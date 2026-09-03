"""Render the training curve as a slide-ready PNG.

    python plot_metrics.py --run demo --dark

Three panels, chosen because together they tell the whole story:

  1. reward per episode      -- the learning curve, including the dips
  2. time spent balanced     -- when it went from "reaches the top" to "stays"
  3. entropy temperature     -- SAC's own exploration dial, annealing itself

The milestone annotations come from the checkpoint evals, not hand-picked, so
the arrows land on whatever actually happened in this run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="demo")
    ap.add_argument("--out", default=None, help="default: runs/<run>/curve.png")
    ap.add_argument("--dark", action="store_true", help="dark background, to match the viewer")
    ap.add_argument("--baseline", type=float, default=None,
                    help="draw a reference line at this return (e.g. the hand-tuned controller)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run
    with (run_dir / "metrics.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    ckpts = sorted(manifest["checkpoints"], key=lambda e: e["episode"])

    ep = np.array([int(r["episode"]) for r in rows])
    ret = np.array([float(r["return"]) for r in rows])
    upright = np.array([float(r["upright_frac"]) for r in rows])
    alpha = np.array([float(r["alpha"]) if r["alpha"] not in ("", "nan") else np.nan for r in rows])

    ink, dim, grid_c = ("#e8ecf4", "#8a94ae", "#2a2f3d") if args.dark else ("#1a1a1a", "#666666", "#dddddd")
    bg = "#10121a" if args.dark else "white"
    accent, good, warn = "#5eb0ff", "#56d694", "#ffb054"

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1, 1], "hspace": 0.16})
    fig.patch.set_facecolor(bg)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, color=grid_c, linewidth=0.6, alpha=0.7)
        ax.tick_params(colors=dim, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_c)

    # --- panel 1: return
    ax = axes[0]
    ax.plot(ep, ret, color=accent, alpha=0.28, linewidth=1, label="episode return")
    ax.plot(ep, moving_average(ret, 10), color=accent, linewidth=2.2, label="10-episode average")
    ce = [c["episode"] for c in ckpts]
    cr = [c["eval_return"] for c in ckpts]
    ax.plot(ce, cr, "o", color=good, markersize=4.5, label="checkpoint (deterministic eval)")
    if args.baseline is not None:
        ax.axhline(args.baseline, color=warn, linestyle="--", linewidth=1.4,
                   label=f"hand-tuned controller ({args.baseline:.0f})")
    ax.set_ylabel("reward per episode", color=ink, fontsize=10)
    ax.set_title(f"Cart-pole swing-up: SAC learning curve  ({args.run})",
                 color=ink, fontsize=13, pad=12, loc="left")
    leg = ax.legend(loc="lower right", fontsize=8.5, facecolor=bg, edgecolor=grid_c, framealpha=0.9)
    for t in leg.get_texts():
        t.set_color(dim)

    # --- milestone arrows, read off the checkpoint evals
    def first(pred):
        return next((c for c in ckpts if pred(c)), None)

    milestones = [
        (first(lambda c: c["eval_min_angle"] < 1.5), "starts swinging"),
        (first(lambda c: c["eval_min_angle"] < 0.25), "reaches upright"),
        (first(lambda c: c["eval_solved"] >= 0.5), "holds it there"),
    ]
    # Park the labels at fixed heights in the empty middle of the panel and
    # let the arrows do the pointing. Offsetting from each point instead makes
    # the late-training labels shoot off the top of the axes.
    top = max(cr) if cr else 1.0
    ax.set_ylim(min(0.0, min(ret) * 1.1), top * 1.08)
    seen: set[int] = set()
    for c, label in milestones:
        if c is None or c["episode"] in seen:
            continue
        tier = len(seen)
        seen.add(c["episode"])
        ax.annotate(
            label, xy=(c["episode"], c["eval_return"]),
            xytext=(c["episode"] + max(ep) * 0.035, top * (0.80 - 0.17 * tier)),
            color=ink, fontsize=8.5,
            arrowprops={"arrowstyle": "->", "color": dim, "linewidth": 1,
                        "shrinkA": 2, "shrinkB": 4},
        )

    # --- panel 2: fraction of the episode spent balanced
    ax = axes[1]
    ax.plot(ep, moving_average(upright, 10) * 100, color=good, linewidth=2)
    ax.set_ylabel("time balanced (%)", color=ink, fontsize=10)
    ax.set_ylim(-3, 103)

    # --- panel 3: learned entropy temperature
    ax = axes[2]
    ax.plot(ep, alpha, color=warn, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_ylabel("entropy temp.  alpha", color=ink, fontsize=10)
    ax.set_xlabel("training episode", color=ink, fontsize=10)

    out = Path(args.out) if args.out else run_dir / ("curve_dark.png" if args.dark else "curve.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=bg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
