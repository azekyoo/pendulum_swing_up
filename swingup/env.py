"""Gym-style environment wrapper around the cart-pole swing-up physics.

Deliberately dependency-free (NumPy only) so the physics can be ported to
JavaScript for the web replay without a Python runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .physics import PhysicsParams, integrate, wrap_angle, energy, TH, THD, X, XD


@dataclass(frozen=True)
class RewardParams:
    """Shaping weights.  The reward is a PRODUCT of three terms in [0, 1] so
    that the velocity and centering penalties are automatically gated by how
    upright the rod is: while swinging up, ``upright`` is small and the
    velocity term barely matters, so the agent is free to spin fast.  Near the
    goal, ``upright`` is ~1 and the velocity term dominates, which is exactly
    where we want it to demand stillness."""

    vel_scale: float = 0.1     # bigger -> harsher angular-velocity penalty
    vel_weight: float = 0.5    # fraction of reward gated by stillness
    center_scale: float = 0.5  # bigger -> harsher off-centre penalty
    action_cost: float = 0.002  # control effort penalty


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 500          # 10 s at 50 Hz
    terminate_on_rail: bool = True  # hitting the rail end ends the episode
    init_angle_noise: float = 0.05  # rad, spread around hanging
    init_x_noise: float = 0.05      # m
    init_vel_noise: float = 0.05    # m/s and rad/s


class CartPoleSwingUpEnv:
    """Continuous-action cart-pole swing-up.

    Action: array of shape (1,) in [-1, 1]; scaled to +/- ``force_mag`` newtons.
            -1 pushes the cart left, +1 right.  (Human keyboard control maps
            the arrow keys to exactly -1 / +1, so a person plays bang-bang
            while the agent gets the full continuous range.)

    Observation (5,): [x / x_limit, x_dot / 5, cos(theta), sin(theta),
                       theta_dot / 10]
            cos/sin instead of raw theta so the network never sees the
            discontinuity at +/-pi, and the /5, /10 divisors keep every input
            roughly in [-1, 1].
    """

    obs_dim = 5
    act_dim = 1

    def __init__(
        self,
        physics: PhysicsParams | None = None,
        config: EnvConfig | None = None,
        reward: RewardParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.p = physics or PhysicsParams()
        self.cfg = config or EnvConfig()
        self.rw = reward or RewardParams()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4)
        self.steps = 0

    # ------------------------------------------------------------------ api

    def reset(self, seed: int | None = None, state: np.ndarray | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if state is not None:
            self.state = np.asarray(state, dtype=np.float64).copy()
        else:
            c = self.cfg
            self.state = np.array(
                [
                    self.rng.normal(0.0, c.init_x_noise),
                    self.rng.normal(0.0, c.init_vel_noise),
                    np.pi + self.rng.normal(0.0, c.init_angle_noise),
                    self.rng.normal(0.0, c.init_vel_noise),
                ]
            )
        self.steps = 0
        return self.observe()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.state = integrate(self.state, a * self.p.force_mag, self.p)
        self.state[TH] = wrap_angle(self.state[TH])
        self.steps += 1

        reward = self.reward(a)
        off_rail = abs(self.state[X]) > self.p.x_limit
        terminated = bool(off_rail and self.cfg.terminate_on_rail)
        truncated = self.steps >= self.cfg.max_steps
        if off_rail and not self.cfg.terminate_on_rail:
            # Bounce off the rail end with no restitution instead of ending.
            self.state[X] = np.clip(self.state[X], -self.p.x_limit, self.p.x_limit)
            self.state[XD] = 0.0

        return self.observe(), reward, terminated, truncated, self.info(off_rail)

    # -------------------------------------------------------------- reward

    def reward(self, action: float) -> float:
        _, _, th, thd = self.state
        x = self.state[X]
        upright = 0.5 * (1.0 + np.cos(th))                       # 0 down, 1 up
        slow = 1.0 / (1.0 + self.rw.vel_scale * thd ** 2)        # 1 when still
        centered = 1.0 / (1.0 + self.rw.center_scale * (x / self.p.x_limit) ** 2)
        gate = (1.0 - self.rw.vel_weight) + self.rw.vel_weight * slow
        return float(upright * gate * centered - self.rw.action_cost * action ** 2)

    # ---------------------------------------------------------------- views

    def observe(self) -> np.ndarray:
        x, xd, th, thd = self.state
        return np.array(
            [x / self.p.x_limit, xd / 5.0, np.cos(th), np.sin(th), thd / 10.0],
            dtype=np.float32,
        )

    def info(self, off_rail: bool = False) -> dict:
        th = wrap_angle(self.state[TH])
        return {
            "x": float(self.state[X]),
            "theta": float(th),
            "angle_from_upright": float(abs(th)),
            "is_upright": bool(abs(th) < 0.2 and abs(self.state[THD]) < 1.0),
            "energy": energy(self.state, self.p),
            "off_rail": bool(off_rail),
        }

    def spec(self) -> dict:
        """Everything the JS renderer needs to draw and re-simulate."""
        return {"physics": self.p.to_dict(), "config": asdict(self.cfg), "reward": asdict(self.rw)}
