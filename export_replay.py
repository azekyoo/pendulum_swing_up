"""Roll out saved checkpoints and dump them as JSON for the web replay page.

    python export_replay.py --run demo            # auto-pick milestone clips
    python export_replay.py --episodes 0 40 70 200 --run demo

By default it picks story beats rather than evenly spaced episodes, because
"the first time it ever touched vertical" is a better slide than "episode 120":

    untrained -> first real swing -> first time upright -> first solved -> best

The output (web/replay.json) is self-contained: physics constants, the full
training curve, and every clip's state trajectory.  The web page replays the
recorded states, so it needs no Python and no neural network at run time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from swingup.baseline import EnergyBaseline
from swingup.env import CartPoleSwingUpEnv, EnvConfig, RewardParams
from swingup.physics import PhysicsParams
from swingup.rollout import actor_policy, rollout, zero_policy
from swingup.sac import SAC

# Seed shared by every clip, so all of them start from the same hanging state
# and the only thing that differs between clips is the policy.
CLIP_SEED = 10_000


def pick_milestones(entries: list[dict]) -> list[dict]:
    """Choose narrative checkpoints: nothing, swinging, touching, solving, best."""
    picked: dict[int, str] = {}

    def take(entry: dict | None, label: str) -> None:
        if entry is not None and entry["episode"] not in picked:
            picked[entry["episode"]] = label

    def first(pred) -> dict | None:
        return next((e for e in entries if pred(e)), None)

    take(entries[0], "untrained")
    take(first(lambda e: e["eval_min_angle"] < 1.5), "starts swinging")
    take(first(lambda e: e["eval_min_angle"] < 0.25), "first time upright")
    take(first(lambda e: e["eval_solved"] >= 0.5), "first balance")
    take(max(entries, key=lambda e: e["eval_return"]), "best")
    take(entries[-1], "final")

    order = {e["episode"]: i for i, e in enumerate(entries)}
    chosen = sorted(picked.items(), key=lambda kv: order[kv[0]])
    by_episode = {e["episode"]: e for e in entries}
    return [{**by_episode[ep], "label": label} for ep, label in chosen]


def load_curve(run_dir: Path) -> list[dict]:
    """Per-episode training returns, thinned so the JSON stays small."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    def num(row: dict, key: str) -> float:
        try:
            return float(row[key])
        except (ValueError, KeyError):
            return 0.0

    return [
        {
            "episode": int(r["episode"]),
            "step": int(r["step"]),
            "return": round(num(r, "return"), 2),
            "upright_frac": round(num(r, "upright_frac"), 4),
            "off_rail": int(num(r, "off_rail")),
        }
        for r in rows
    ]


def clip_from_rollout(result: dict, label: str, kind: str, extra: dict) -> dict:
    return {
        "label": label,
        "kind": kind,
        "frames": result["frames"],
        "actions": result["actions"],
        "rewards": result["rewards"],
        "return": round(result["return"], 2),
        "upright_frac": round(result["upright_frac"], 4),
        "min_angle": round(result["min_angle"], 4),
        "off_rail": result["off_rail"],
        "solved": result["solved"],
        **extra,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="demo")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="explicit episodes to export (default: auto milestones)")
    ap.add_argument("--out", default="web/replay.json")
    ap.add_argument("--no-baseline", action="store_true", help="skip the classical-controller clip")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run
    manifest = json.loads((run_dir / "manifest.json").read_text())
    entries = sorted(manifest["checkpoints"], key=lambda e: e["episode"])
    spec = manifest["spec"]

    env = CartPoleSwingUpEnv(
        physics=PhysicsParams(**spec["physics"]),
        config=EnvConfig(**spec["config"]),
        reward=RewardParams(**spec["reward"]),
    )

    if args.episodes:
        by_episode = {e["episode"]: e for e in entries}
        chosen = []
        for ep in args.episodes:
            nearest = min(entries, key=lambda e: abs(e["episode"] - ep))
            chosen.append({**by_episode[nearest["episode"]], "label": f"episode {nearest['episode']}"})
    else:
        chosen = pick_milestones(entries)

    clips = []
    for entry in chosen:
        actor = SAC.actor_only(run_dir / "checkpoints" / entry["file"])[0]
        result = rollout(env, actor_policy(actor), seed=CLIP_SEED, record=True)
        clips.append(clip_from_rollout(
            result, entry["label"], "agent",
            {"episode": entry["episode"], "step": entry["step"],
             "eval_return": round(entry["eval_return"], 2)},
        ))
        print(f"ep {entry['episode']:4d}  {entry['label']:<18} return {result['return']:6.1f}  "
              f"upright {result['upright_frac']:.0%}  min_angle {np.degrees(result['min_angle']):5.1f} deg")

    if not args.no_baseline:
        result = rollout(env, EnergyBaseline(env), seed=CLIP_SEED, record=True)
        clips.append(clip_from_rollout(result, "hand-tuned controller", "baseline",
                                      {"episode": None, "step": None, "eval_return": None}))
        print(f"      {'baseline':<18} return {result['return']:6.1f}  upright {result['upright_frac']:.0%}")

    result = rollout(env, zero_policy, seed=CLIP_SEED, record=True)
    clips.append(clip_from_rollout(result, "no control", "none",
                                  {"episode": None, "step": None, "eval_return": None}))

    payload = {
        "run": args.run,
        "spec": spec,
        "curve": load_curve(run_dir),
        "checkpoints": [
            {k: e[k] for k in ("episode", "step", "eval_return", "eval_upright_frac", "eval_solved")}
            for e in entries
        ],
        "clips": clips,
    }
    blob = json.dumps(payload, separators=(",", ":"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob)
    # Also emit a JS-wrapped copy: browsers block fetch() over file://, but a
    # <script src> tag works, so web/index.html opens by double-click.
    js = out.with_suffix(".js")
    js.write_text(f"window.REPLAY_DATA = {blob};\n")
    print(f"\nwrote {out} and {js}  ({out.stat().st_size / 1024:.0f} KB, {len(clips)} clips)")


if __name__ == "__main__":
    main()
