"""Physics tests.

The point of these is that an RL agent will happily exploit a broken
simulator: if the integrator quietly manufactures energy, the policy learns to
farm that instead of solving the task, and the bug only shows up as a policy
that looks great and behaves absurdly.  So the invariants get checked.
"""

import numpy as np
import pytest

from swingup.env import CartPoleSwingUpEnv, EnvConfig
from swingup.physics import (
    PhysicsParams, X, XD, TH, THD, derivative, energy, integrate, wrap_angle,
)

FRICTIONLESS = PhysicsParams(b_cart=0.0, b_pole=0.0)


def momentum(state, p):
    """Total horizontal momentum; conserved when no external force acts."""
    _, xd, th, thd = state
    return (p.m_cart + p.m_pole) * xd + p.m_pole * p.half_length * np.cos(th) * thd


def test_energy_conserved_without_damping():
    state = np.array([0.0, 0.0, np.pi - 0.4, 0.0])
    e0 = energy(state, FRICTIONLESS)
    for _ in range(2500):  # 50 s
        state = integrate(state, 0.0, FRICTIONLESS)
    assert abs(energy(state, FRICTIONLESS) - e0) < 1e-5


def test_horizontal_momentum_conserved_without_force():
    state = np.array([0.0, 0.3, 2.0, -1.0])
    m0 = momentum(state, FRICTIONLESS)
    for _ in range(1000):
        state = integrate(state, 0.0, FRICTIONLESS)
    assert abs(momentum(state, FRICTIONLESS) - m0) < 1e-6


def test_damping_only_removes_energy():
    p = PhysicsParams()
    state = np.array([0.0, 0.0, np.pi / 2, 0.0])
    previous = energy(state, p)
    for _ in range(500):
        state = integrate(state, 0.0, p)
        current = energy(state, p)
        assert current <= previous + 1e-9
        previous = current


def test_hanging_is_stable_upright_is_not():
    p = PhysicsParams()
    hanging = np.array([0.0, 0.0, np.pi - 0.02, 0.0])
    upright = np.array([0.0, 0.0, 0.02, 0.0])
    # Angular acceleration should push back towards pi, and away from 0.
    assert derivative(hanging, 0.0, p)[THD] > 0      # theta grows back to pi
    assert derivative(upright, 0.0, p)[THD] > 0      # theta grows away from 0


def test_rest_states_are_equilibria():
    p = PhysicsParams()
    for th in (0.0, np.pi):
        d = derivative(np.array([0.0, 0.0, th, 0.0]), 0.0, p)
        assert abs(d[XD]) < 1e-12 and abs(d[THD]) < 1e-12


def test_pushing_right_accelerates_cart_right():
    p = PhysicsParams()
    d = derivative(np.array([0.0, 0.0, np.pi, 0.0]), p.force_mag, p)
    assert d[XD] > 0


def test_integrator_matches_finer_timestep():
    """RK4 at the control rate should agree with a 10x finer simulation."""
    coarse = PhysicsParams(b_cart=0.0, b_pole=0.0, substeps=2)
    fine = PhysicsParams(b_cart=0.0, b_pole=0.0, substeps=20)
    a = b = np.array([0.0, 0.0, np.pi - 0.3, 0.0])
    for _ in range(250):
        a = integrate(a, 5.0, coarse)
        b = integrate(b, 5.0, fine)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("raw", [0.0, 1.2, np.pi, 3 * np.pi, -3.5 * np.pi, 12.0, -40.0])
def test_wrap_angle(raw):
    wrapped = wrap_angle(raw)
    assert -np.pi <= wrapped < np.pi
    # Same physical direction: differs from the input by a whole turn.
    turns = (raw - wrapped) / (2 * np.pi)
    assert abs(turns - round(turns)) < 1e-9


def test_swinging_up_requires_more_than_one_push():
    """Sanity check that the force limit really does forbid a single-shot lift:
    holding full force from rest must not reach vertical."""
    env = CartPoleSwingUpEnv(config=EnvConfig(terminate_on_rail=False))
    env.reset(state=np.array([0.0, 0.0, np.pi, 0.0]))
    best = np.pi
    for _ in range(500):
        _, _, _, _, info = env.step(1.0)
        best = min(best, info["angle_from_upright"])
    assert best > 0.2, "a constant push should not be able to balance it"


# ---------------------------------------------------------------- environment


def test_reward_is_maximal_upright_and_still():
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([0.0, 0.0, 0.0, 0.0]))
    assert env.reward(0.0) == pytest.approx(1.0)


def test_reward_is_near_zero_hanging():
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([0.0, 0.0, np.pi, 0.0]))
    assert env.reward(0.0) < 1e-6


def test_reward_prefers_still_over_spinning_at_the_top():
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([0.0, 0.0, 0.0, 0.0]))
    still = env.reward(0.0)
    env.reset(state=np.array([0.0, 0.0, 0.0, 6.0]))
    spinning = env.reward(0.0)
    assert still > spinning


def test_velocity_penalty_is_gated_by_uprightness():
    """While hanging, spinning fast must cost almost nothing, or the agent
    never learns to pump energy in the first place."""
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([0.0, 0.0, np.pi, 0.0]))
    slow = env.reward(0.0)
    env.reset(state=np.array([0.0, 0.0, np.pi, 8.0]))
    fast = env.reward(0.0)
    assert abs(slow - fast) < 0.01


def test_observation_is_bounded_and_continuous_across_pi():
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([0.0, 0.0, np.pi - 1e-6, 0.0]))
    a = env.observe()
    env.reset(state=np.array([0.0, 0.0, -np.pi + 1e-6, 0.0]))
    b = env.observe()
    assert np.allclose(a, b, atol=1e-4)   # no discontinuity at the wrap point
    assert np.all(np.abs(a) <= 1.5)


def test_episode_terminates_when_cart_leaves_the_rail():
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([2.35, 0.0, np.pi, 0.0]))
    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(1.0)
        if terminated or truncated:
            break
    assert terminated and info["off_rail"]


def test_episode_truncates_at_max_steps():
    env = CartPoleSwingUpEnv(config=EnvConfig(max_steps=25))
    env.reset()
    for i in range(25):
        _, _, terminated, truncated, _ = env.step(0.0)
    assert truncated and not terminated and env.steps == 25


def test_reset_seed_is_reproducible():
    env = CartPoleSwingUpEnv()
    a = env.reset(seed=7)
    b = env.reset(seed=7)
    assert np.allclose(a, b)


def test_action_is_clipped():
    env = CartPoleSwingUpEnv()
    env.reset(state=np.array([0.0, 0.0, np.pi, 0.0]))
    huge = env.step(1000.0)
    env.reset(state=np.array([0.0, 0.0, np.pi, 0.0]))
    one = env.step(1.0)
    assert np.allclose(huge[0], one[0])
