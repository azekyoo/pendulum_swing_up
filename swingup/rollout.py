"""Shared rollout + evaluation helpers.

Used by training (periodic eval), the pygame viewer, and the replay exporter,
so all three drive the environment through exactly the same code path and a
checkpoint that scores well in training looks identical on screen.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .env import CartPoleSwingUpEnv


class Policy(Protocol):
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


def zero_policy(obs: np.ndarray) -> np.ndarray:
    """Do-nothing baseline: the rod just hangs."""
    return np.zeros(1, dtype=np.float32)


def rollout(
    env: CartPoleSwingUpEnv,
    policy: Policy,
    seed: int | None = None,
    init_state: np.ndarray | None = None,
    record: bool = False,
) -> dict:
    """Run one episode and return summary stats (plus the trajectory if asked).

    ``upright_steps`` counts steps within 0.2 rad of vertical AND under
    1 rad/s, i.e. genuinely balanced rather than merely passing through.
    """
    obs = env.reset(seed=seed, state=init_state)
    total_reward = 0.0
    upright_steps = 0
    min_angle = np.pi
    frames: list[list[float]] = []
    actions: list[float] = []
    rewards: list[float] = []
    terminated = truncated = False
    info = env.info()

    while not (terminated or truncated):
        if record:
            frames.append([round(float(v), 5) for v in env.state])
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        upright_steps += int(info["is_upright"])
        min_angle = min(min_angle, info["angle_from_upright"])
        if record:
            actions.append(round(float(np.asarray(action).reshape(-1)[0]), 5))
            rewards.append(round(reward, 5))

    if record:
        frames.append([round(float(v), 5) for v in env.state])

    out = {
        "return": float(total_reward),
        "steps": env.steps,
        "upright_steps": upright_steps,
        "upright_frac": upright_steps / max(env.steps, 1),
        "min_angle": float(min_angle),
        "off_rail": bool(info["off_rail"]),
        # "Solved" = balanced for the last ~2 s of a full-length episode.
        "solved": bool(upright_steps >= 100 and not info["off_rail"]),
    }
    if record:
        out["frames"] = frames
        out["actions"] = actions
        out["rewards"] = rewards
    return out


def evaluate(
    env: CartPoleSwingUpEnv,
    policy: Policy,
    episodes: int = 3,
    base_seed: int = 10_000,
) -> dict:
    """Deterministic evaluation over fixed seeds, so the numbers logged at
    episode 50 and episode 500 are directly comparable."""
    runs = [rollout(env, policy, seed=base_seed + i) for i in range(episodes)]
    return {
        "eval_return": float(np.mean([r["return"] for r in runs])),
        "eval_return_std": float(np.std([r["return"] for r in runs])),
        "eval_upright_frac": float(np.mean([r["upright_frac"] for r in runs])),
        "eval_min_angle": float(np.mean([r["min_angle"] for r in runs])),
        "eval_solved": float(np.mean([r["solved"] for r in runs])),
    }


def actor_policy(actor, deterministic: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a loaded SAC actor as a plain obs -> action function."""

    def policy(obs: np.ndarray) -> np.ndarray:
        return actor.act(obs, deterministic=deterministic)

    return policy
