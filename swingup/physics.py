"""Cart-pole swing-up physics.

A rigid rod (mass ``m``, full length ``L``, half-length ``l = L/2``) is pivoted
on a cart (mass ``M``) that slides along a horizontal rail.  The only actuator
is a horizontal force on the cart; the hinge is free.  The agent must pump
energy into the rod by moving the cart, then catch and hold it upright.

Angle convention
----------------
``th`` is measured from the UPRIGHT vertical, positive counter-clockwise.

    th = 0      -> upright (goal, unstable)
    th = pi     -> hanging straight down (rest, stable)

Equations of motion
-------------------
From the Lagrangian, with ``J = m l**2 + I_com = (4/3) m l**2`` the rod's
moment of inertia about the pivot:

    T = 1/2 (M+m) xd**2  +  m l cos(th) xd thd  +  1/2 J thd**2
    V = m g l cos(th)

Adding viscous damping ``-b_cart * xd`` on the cart and ``-b_pole * thd`` on
the hinge yields a coupled 2x2 linear system in the accelerations:

    | M+m          m l cos(th) | | xdd  |   | F - b_cart xd + m l sin(th) thd**2 |
    | m l cos(th)  J           | | thdd | = | m g l sin(th) - b_pole thd         |

which is solved in closed form below.  This is exact for all angles (no small
angle approximation), so the same equations cover swing-up and balancing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# State vector layout: [x, x_dot, theta, theta_dot]
X, XD, TH, THD = 0, 1, 2, 3


@dataclass(frozen=True)
class PhysicsParams:
    """Physical constants.  Defaults are a swing-up-feasible configuration:
    the force limit is too small to whip the rod up in one shot, so the agent
    is forced to learn resonant energy pumping."""

    gravity: float = 9.81
    m_cart: float = 0.5      # kg
    m_pole: float = 0.5      # kg
    pole_length: float = 1.2  # m, full rod length
    force_mag: float = 15.0   # N, applied at |action| == 1
    b_cart: float = 0.05      # N/(m/s), rail friction
    b_pole: float = 0.02      # N.m/(rad/s), hinge friction + air drag
    dt: float = 0.02          # s, control period (50 Hz)
    substeps: int = 2         # RK4 sub-steps per control step
    x_limit: float = 2.4      # m, rail half-length

    @property
    def half_length(self) -> float:
        return 0.5 * self.pole_length

    @property
    def inertia_pivot(self) -> float:
        """J: rod inertia about the pivot = (4/3) m l**2."""
        return (4.0 / 3.0) * self.m_pole * self.half_length ** 2

    def to_dict(self) -> dict:
        return asdict(self)


def derivative(state: np.ndarray, force: float, p: PhysicsParams) -> np.ndarray:
    """Time derivative of ``state`` under horizontal cart ``force`` (Newtons)."""
    _, xd, th, thd = state
    m, l, g = p.m_pole, p.half_length, p.gravity
    total_mass = p.m_cart + m
    ml = m * l
    cos_th, sin_th = np.cos(th), np.sin(th)

    # Right-hand side of the 2x2 system.
    a = force - p.b_cart * xd + ml * sin_th * thd ** 2   # cart equation
    b = m * g * l * sin_th - p.b_pole * thd              # pole equation

    # Cramer's rule.  det > 0 always: (M+m)J - (ml cos)^2 >= (M+m)(4/3)ml^2 - (ml)^2.
    coupling = ml * cos_th
    det = total_mass * p.inertia_pivot - coupling ** 2
    xdd = (p.inertia_pivot * a - coupling * b) / det
    thdd = (total_mass * b - coupling * a) / det

    return np.array([xd, xdd, thd, thdd], dtype=np.float64)


def rk4_step(state: np.ndarray, force: float, dt: float, p: PhysicsParams) -> np.ndarray:
    """One classical Runge-Kutta 4 step.

    RK4 rather than Euler because swing-up is an energy-shaping task: Euler's
    per-step energy drift lets the agent exploit fake energy and the learned
    policy then fails on any finer integrator.
    """
    k1 = derivative(state, force, p)
    k2 = derivative(state + 0.5 * dt * k1, force, p)
    k3 = derivative(state + 0.5 * dt * k2, force, p)
    k4 = derivative(state + dt * k3, force, p)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(state: np.ndarray, force: float, p: PhysicsParams) -> np.ndarray:
    """Advance one control period, holding ``force`` constant (zero-order hold)."""
    sub_dt = p.dt / p.substeps
    for _ in range(p.substeps):
        state = rk4_step(state, force, sub_dt, p)
    return state


def energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Total mechanical energy.  Conserved when force and damping are zero;
    used by the tests as an integrator sanity check."""
    _, xd, th, thd = state
    m, l = p.m_pole, p.half_length
    kinetic = (
        0.5 * (p.m_cart + m) * xd ** 2
        + m * l * np.cos(th) * xd * thd
        + 0.5 * p.inertia_pivot * thd ** 2
    )
    potential = m * p.gravity * l * np.cos(th)
    return float(kinetic + potential)


def wrap_angle(th: float | np.ndarray) -> float | np.ndarray:
    """Wrap to the half-open interval [-pi, pi), so ``abs(th)`` is the true
    distance from upright.  Note that exactly pi comes back as -pi; the two
    describe the same physical angle, and every consumer here uses either
    ``abs`` or ``cos``/``sin``, so the choice of endpoint never matters."""
    return (th + np.pi) % (2.0 * np.pi) - np.pi
