"""Hand-written classical controller, for comparison against the learned one.

This is the textbook two-mode solution and it exists here purely so the talk
can contrast it with RL:

  * it needs the equations of motion, the mass, the length and gravity;
  * it needs a human to pick five gains and one switching threshold;
  * it needs someone to notice that swing-up and balancing are different
    problems requiring different controllers.

SAC needs none of that -- only the reward. That is the whole point of the
comparison, not that one is faster than the other.

Mode 1, energy pumping (Astrom & Furuta).  For the rod alone,

    dE/dt = -m * l * xdd * thd * cos(th)

so choosing a cart acceleration opposed in sign to ``thd * cos(th)`` always
adds energy.  Scaling by the energy deficit makes the push vanish as the rod
reaches the energy of the upright position.

Mode 2, stabilisation.  Near vertical the system linearises, so a fixed
linear feedback on all four state variables holds it there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import CartPoleSwingUpEnv
from .physics import wrap_angle


@dataclass(frozen=True)
class BaselineGains:
    k_energy: float = 2.0      # energy-pumping gain
    k_x: float = 0.60          # cart centring during pumping
    k_xd: float = 0.55         # cart damping during pumping
    switch_angle: float = 0.6  # rad from upright at which we switch to balance
    # Balance-mode feedback on [theta, theta_dot, x, x_dot]
    balance: tuple[float, float, float, float] = (7.0, 1.4, 0.45, 0.9)
    kick: float = 1.0          # opening push, since pumping stalls at exact rest


class EnergyBaseline:
    """Callable ``obs -> action``, matching the Policy protocol in rollout.py.

    Reads the true state off the env rather than the observation, because a
    classical controller is assumed to have a calibrated model. The learned
    policy gets only the 5-number observation.
    """

    def __init__(self, env: CartPoleSwingUpEnv, gains: BaselineGains | None = None):
        self.env = env
        self.g = gains or BaselineGains()

    def pole_energy(self) -> float:
        """Rod energy relative to 'upright and motionless', so 0 is the target
        and the hanging rest state is -2*m*g*l."""
        _, _, th, thd = self.env.state
        p = self.env.p
        return float(
            0.5 * p.inertia_pivot * thd ** 2
            + p.m_pole * p.gravity * p.half_length * (np.cos(th) - 1.0)
        )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x, xd, th, thd = self.env.state
        th = wrap_angle(th)
        g = self.g

        if abs(th) < g.switch_angle:
            kth, kthd, kx, kxd = g.balance
            u = kth * th + kthd * thd + kx * x + kxd * xd
        else:
            energy_deficit = self.pole_energy()
            u = g.k_energy * energy_deficit * thd * np.cos(th) - g.k_x * x - g.k_xd * xd
            if abs(thd) < 0.05 and abs(energy_deficit) > 1.0:
                u = g.kick  # break out of dead rest, where the pumping term is 0

        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)
