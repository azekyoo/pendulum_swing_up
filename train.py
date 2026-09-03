"""Train SAC on cart-pole swing-up, checkpointing often enough to replay the
whole learning curve afterwards.

    python train.py --episodes 300 --run demo

Writes everything under runs/<run>/:

    config.json      physics, reward and SAC settings for this run
    metrics.csv      one row per training episode
    checkpoints/     ep00000.pt, ep00010.pt, ... plus latest.pt
    manifest.json    checkpoint index with each one's eval score

The checkpoint at episode 0 is saved BEFORE any learning, so the replay always
starts from a genuinely untrained policy -- that "random flailing" clip is the
before half of the before/after.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from swingup.env import CartPoleSwingUpEnv, EnvConfig, RewardParams
from swingup.physics import PhysicsParams
from swingup.rollout import actor_policy, evaluate, rollout
from swingup.sac import SAC, SACConfig

METRIC_FIELDS = [
    "episode", "step", "return", "length", "upright_steps", "upright_frac",
    "min_angle", "off_rail", "solved", "critic_loss", "actor_loss", "alpha",
    "entropy", "q_mean", "wall_time",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=300, help="training episodes (500 steps each)")
    ap.add_argument("--run", default="demo", help="run name under runs/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10, help="episodes between checkpoints")
    ap.add_argument("--eval-episodes", type=int, default=3, help="deterministic eval episodes per checkpoint")
    ap.add_argument("--force-mag", type=float, default=None, help="override cart force limit (N)")
    ap.add_argument("--max-steps", type=int, default=500, help="episode length in steps")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--resume", action="store_true", help="continue from runs/<run>/checkpoints/latest.pt")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)

    physics = PhysicsParams(**({"force_mag": args.force_mag} if args.force_mag else {}))
    env_cfg = EnvConfig(max_steps=args.max_steps)
    reward = RewardParams()
    env = CartPoleSwingUpEnv(physics=physics, config=env_cfg, reward=reward, seed=args.seed)
    eval_env = CartPoleSwingUpEnv(physics=physics, config=env_cfg, reward=reward)

    agent = SAC(env.obs_dim, env.act_dim, SACConfig(seed=args.seed))

    run_dir = Path("runs") / args.run
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**env.spec(), "sac": agent.cfg.to_dict(), "args": vars(args)}, indent=2)
    )

    manifest: list[dict] = []
    start_episode = 0
    if args.resume and (ckpt_dir / "latest.pt").exists():
        ckpt = agent.load(ckpt_dir / "latest.pt")
        start_episode = int(ckpt["extra"].get("episode", 0)) + 1
        manifest = json.loads((run_dir / "manifest.json").read_text())["checkpoints"]
        print(f"resumed from episode {start_episode}, {agent.total_steps} steps")

    metrics_path = run_dir / "metrics.csv"
    new_file = not metrics_path.exists() or not args.resume
    metrics_file = metrics_path.open("w" if new_file else "a", newline="")
    writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
    if new_file:
        writer.writeheader()

    def checkpoint(episode: int) -> dict:
        """Eval the current deterministic policy and write a checkpoint."""
        stats = evaluate(eval_env, actor_policy(agent.actor), episodes=args.eval_episodes)
        name = f"ep{episode:05d}.pt"
        extra = {"episode": episode, **stats}
        agent.save(ckpt_dir / name, extra=extra)
        agent.save(ckpt_dir / "latest.pt", extra=extra)
        entry = {"episode": episode, "step": agent.total_steps, "file": name, **stats}
        manifest.append(entry)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run": args.run, "spec": env.spec(), "checkpoints": manifest}, indent=2)
        )
        return entry

    print(f"run={args.run}  force={physics.force_mag}N  rail=+/-{physics.x_limit}m  "
          f"{args.episodes} episodes x {args.max_steps} steps  threads={args.threads}")

    # Snapshot the untrained policy first: this is the "before" clip.
    if start_episode == 0:
        e = checkpoint(0)
        print(f"ep     0 | UNTRAINED baseline: eval_return={e['eval_return']:6.1f} "
              f"upright={e['eval_upright_frac']:.2f}")

    t0 = time.time()
    recent: list[float] = []
    for episode in range(start_episode, args.episodes + 1):
        obs = env.reset()
        ep_return = 0.0
        upright_steps = 0
        min_angle = np.pi
        losses: dict[str, float] = {}
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Only a real terminal (rail crash) stops the bootstrap; a timeout
            # does not, or the agent would learn the episode clock is a cliff.
            agent.buffer.add(obs, action, reward, next_obs, terminated)
            agent.total_steps += 1
            obs = next_obs
            ep_return += reward
            upright_steps += int(info["is_upright"])
            min_angle = min(min_angle, info["angle_from_upright"])
            for _ in range(agent.cfg.updates_per_step):
                out = agent.update()
                if out:
                    losses = out

        recent.append(ep_return)
        recent = recent[-20:]
        writer.writerow({
            "episode": episode,
            "step": agent.total_steps,
            "return": round(ep_return, 3),
            "length": env.steps,
            "upright_steps": upright_steps,
            "upright_frac": round(upright_steps / max(env.steps, 1), 4),
            "min_angle": round(float(min_angle), 4),
            "off_rail": int(info["off_rail"]),
            "solved": int(upright_steps >= 100 and not info["off_rail"]),
            "critic_loss": round(losses.get("critic_loss", float("nan")), 5),
            "actor_loss": round(losses.get("actor_loss", float("nan")), 5),
            "alpha": round(losses.get("alpha", float("nan")), 5),
            "entropy": round(losses.get("entropy", float("nan")), 5),
            "q_mean": round(losses.get("q_mean", float("nan")), 5),
            "wall_time": round(time.time() - t0, 2),
        })
        metrics_file.flush()

        if episode % args.ckpt_every == 0 and episode > 0:
            e = checkpoint(episode)
            print(f"ep {episode:5d} | step {agent.total_steps:7d} | train {np.mean(recent):6.1f} "
                  f"| eval {e['eval_return']:6.1f} +/-{e['eval_return_std']:5.1f} "
                  f"| upright {e['eval_upright_frac']:.2f} | solved {e['eval_solved']:.0%} "
                  f"| alpha {losses.get('alpha', 0):.3f} | {time.time() - t0:5.0f}s")

    metrics_file.close()
    best = max(manifest, key=lambda e: e["eval_return"])
    print(f"\ndone in {time.time() - t0:.0f}s. {len(manifest)} checkpoints in {ckpt_dir}")
    print(f"best: episode {best['episode']} eval_return={best['eval_return']:.1f} "
          f"upright={best['eval_upright_frac']:.2f}")


if __name__ == "__main__":
    main()
