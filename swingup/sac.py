"""Soft Actor-Critic for continuous cart-pole swing-up.

Why SAC rather than DQN or PPO here: swing-up needs (a) fine, smooth force
near the balance point, which a 3-way left/none/right discretisation handles
badly, and (b) sample efficiency, since the whole training run should finish
on a CPU in minutes.  SAC's entropy bonus also supplies the exploration needed
to stumble onto the first successful swing, which is the hard part of the task.

Deliberately one readable file with no RL library, so every line of the
algorithm can be pointed at during the talk.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class SACConfig:
    hidden: tuple[int, ...] = (128, 128)
    gamma: float = 0.99          # discount; 0.99 at 50 Hz is a ~2 s horizon
    tau: float = 0.005           # polyak rate for the target critics
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 400_000
    start_steps: int = 2_000     # uniform-random actions to seed the buffer
    updates_per_step: int = 1
    target_entropy: float | None = None  # defaults to -act_dim
    seed: int = 0

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def mlp(sizes: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers += [nn.Linear(sizes[-1], out_dim)]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Outputs a tanh-squashed Gaussian, so actions always land in [-1, 1]
    while staying reparameterisable, which the SAC actor gradient requires."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        # Change-of-variables correction for the tanh squash.
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            mean, _ = self(t)
            return torch.tanh(mean).squeeze(0).numpy()
        action, _ = self.sample(t)
        return action.squeeze(0).numpy()


class TwinCritic(nn.Module):
    """Two independent Q networks.  The min of the pair is used in the TD
    target, which is what stops SAC over-estimating Q and diverging."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden], 1)
        self.q2 = mlp([obs_dim + act_dim, *hidden], 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        oa = torch.cat([obs, act], dim=-1)
        return self.q1(oa), self.q2(oa)


class ReplayBuffer:
    """Flat preallocated NumPy ring buffer; off-policy replay is what lets one
    lucky early swing-up be learned from thousands of times."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(arr[idx])
            for arr in (self.obs, self.act, self.rew, self.next_obs, self.done)
        )


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig | None = None):
        self.cfg = cfg or SACConfig()
        c = self.cfg
        torch.manual_seed(c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.actor = SquashedGaussianActor(obs_dim, act_dim, tuple(c.hidden))
        self.critic = TwinCritic(obs_dim, act_dim, tuple(c.hidden))
        self.critic_target = TwinCritic(obs_dim, act_dim, tuple(c.hidden))
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=c.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=c.lr_critic)

        # Temperature is learned, so exploration noise is never hand-tuned.
        self.target_entropy = c.target_entropy if c.target_entropy is not None else -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=c.lr_alpha)

        self.buffer = ReplayBuffer(obs_dim, act_dim, c.buffer_size)
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.total_steps < self.cfg.start_steps:
            return self.rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
        return self.actor.act(obs, deterministic=deterministic)

    def update(self) -> dict[str, float]:
        c = self.cfg
        if self.buffer.size < max(c.batch_size, c.start_steps):
            return {}
        obs, act, rew, next_obs, done = self.buffer.sample(c.batch_size, self.rng)

        # --- critic: TD target = r + gamma * (min twin target Q - alpha * logpi)
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            target_v = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = rew + c.gamma * (1.0 - done) * target_v

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt_critic.step()

        # --- actor: maximise Q - alpha * log_pi
        new_act, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_act)
        actor_loss = (self.alpha.detach() * logp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        # --- temperature: pull policy entropy towards target_entropy
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1.0 - c.tau).add_(c.tau * p)

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "entropy": float(-logp.mean().detach()),
            "q_mean": float(q1.mean().detach()),
        }

    # ------------------------------------------------------------ persistence

    def save(self, path, extra: dict | None = None) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
                "sac_config": self.cfg.to_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "total_steps": self.total_steps,
                "extra": extra or {},
            },
            path,
        )

    def load(self, path, full: bool = True) -> dict:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        if full and "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt["critic_target"])
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"])
        self.total_steps = ckpt.get("total_steps", 0)
        return ckpt

    @staticmethod
    def actor_only(path) -> tuple[SquashedGaussianActor, dict]:
        """Load just the policy, for rendering and replay export."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hidden = tuple(ckpt["sac_config"]["hidden"])
        actor = SquashedGaussianActor(ckpt["obs_dim"], ckpt["act_dim"], hidden)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        return actor, ckpt
