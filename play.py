"""Interactive cart-pole swing-up viewer.

    python play.py                    # human control, demo run's checkpoints
    python play.py --run demo --mode agent --episode 200

Controls
--------
    LEFT / RIGHT   push the cart (human mode)
    TAB            cycle control: HUMAN -> AGENT -> BASELINE
    [  ]           previous / next checkpoint
    1 .. 9         jump to 10% .. 90% through training
    0              jump to the untrained checkpoint
    E              jump to the best checkpoint by eval score
    R              reset the episode
    SPACE          pause
    - / =          slow down / speed up
    T              toggle the pole-tip trail
    P              poke the rod (random impulse)
    , / .          shorten / lengthen the rod, live
    ESC / Q        quit

Try HUMAN first: swinging it up by hand makes it obvious why this is hard,
which is what makes the trained policy land.

Two keys worth saving for the end of a demo:
  P  shoves the balanced rod.  Nobody wrote recovery behaviour; it falls out
     of the policy having been rewarded for being upright from every state it
     ever stumbled into.
  .  lengthens the rod the policy trained on.  It degrades and then fails,
     which is the sim-to-real gap in one keystroke.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame

from swingup.baseline import EnergyBaseline
from swingup.env import CartPoleSwingUpEnv, EnvConfig, RewardParams
from swingup.physics import PhysicsParams, wrap_angle
from swingup.rollout import actor_policy
from swingup.sac import SAC

WIDTH, HEIGHT = 1120, 660
GROUND_Y = 430

BG = (16, 18, 24)
PANEL = (24, 27, 36)
INK = (232, 236, 244)
DIM = (128, 138, 158)
ACCENT = (94, 176, 255)
GOOD = (86, 214, 148)
WARN = (255, 176, 84)
BAD = (240, 98, 110)
RAIL = (58, 64, 80)

MODES = ("HUMAN", "AGENT", "BASELINE")


# ----------------------------------------------------------------- checkpoints


class CheckpointLibrary:
    """Lazy-loading index over one run's saved policies."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"no manifest at {manifest_path}\nRun training first:  python train.py --run {run_dir.name}"
            )
        manifest = json.loads(manifest_path.read_text())
        self.spec = manifest["spec"]
        self.entries = sorted(manifest["checkpoints"], key=lambda e: e["episode"])
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def actor(self, index: int):
        if index not in self._cache:
            path = self.run_dir / "checkpoints" / self.entries[index]["file"]
            self._cache[index] = SAC.actor_only(path)[0]
        return self._cache[index]

    def nearest(self, episode: int) -> int:
        return min(range(len(self.entries)), key=lambda i: abs(self.entries[i]["episode"] - episode))

    def best(self) -> int:
        return max(range(len(self.entries)), key=lambda i: self.entries[i]["eval_return"])

    def fraction(self, frac: float) -> int:
        return self.nearest(int(frac * self.entries[-1]["episode"]))


# ---------------------------------------------------------------------- render


class Renderer:
    def __init__(self, env: CartPoleSwingUpEnv):
        self.env = env
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pendulum Swing-Up")
        self.font = pygame.font.SysFont("consolas", 17)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 27, bold=True)
        # Fit the rail plus a margin into the window width.
        self.px_per_m = (WIDTH - 160) / (2 * env.p.x_limit + 1.0)

    def sx(self, x: float) -> int:
        return int(WIDTH / 2 + x * self.px_per_m)

    def sy(self, y: float) -> int:
        return int(GROUND_Y - y * self.px_per_m)

    def draw(self, state: np.ndarray, action: float, hud: dict, trail: list[tuple[float, float]]) -> None:
        s = self.screen
        s.fill(BG)
        p = self.env.p
        x, xd, th, thd = state
        th = wrap_angle(th)

        # --- rail, end stops, centre mark, goal line
        pygame.draw.line(s, RAIL, (self.sx(-p.x_limit), GROUND_Y + 26), (self.sx(p.x_limit), GROUND_Y + 26), 4)
        for end in (-p.x_limit, p.x_limit):
            pygame.draw.line(s, BAD, (self.sx(end), GROUND_Y - 6), (self.sx(end), GROUND_Y + 44), 3)
        pygame.draw.line(s, (36, 40, 52), (self.sx(0), GROUND_Y + 14), (self.sx(0), GROUND_Y + 38), 2)
        goal_y = self.sy(p.pole_length)
        pygame.draw.line(s, (40, 70, 58), (self.sx(-p.x_limit), goal_y), (self.sx(p.x_limit), goal_y), 1)
        s.blit(self.small.render("upright", True, (52, 96, 78)), (self.sx(p.x_limit) - 54, goal_y - 18))

        # --- pole tip trail (shows the energy build-up as widening arcs)
        if len(trail) > 1:
            pts = [(self.sx(tx), self.sy(ty)) for tx, ty in trail]
            pygame.draw.lines(s, (44, 62, 88), False, pts, 2)

        # --- cart
        cart_w, cart_h = int(0.62 * self.px_per_m), int(0.26 * self.px_per_m)
        cart = pygame.Rect(0, 0, cart_w, cart_h)
        cart.center = (self.sx(x), GROUND_Y)
        pygame.draw.rect(s, (66, 74, 94), cart, border_radius=5)
        pygame.draw.rect(s, ACCENT if abs(action) > 0.02 else DIM, cart, width=2, border_radius=5)
        for wheel_dx in (-cart_w // 3, cart_w // 3):
            pygame.draw.circle(s, (40, 46, 60), (cart.centerx + wheel_dx, cart.bottom + 6), 7)

        # --- applied force arrow
        if abs(action) > 0.02:
            length = int(action * 0.9 * self.px_per_m)
            y = GROUND_Y + 2
            pygame.draw.line(s, WARN, (cart.centerx, y), (cart.centerx + length, y), 5)
            sign = 1 if length > 0 else -1
            tip = cart.centerx + length
            pygame.draw.polygon(s, WARN, [(tip + sign * 11, y), (tip, y - 7), (tip, y + 7)])

        # --- pole: tip = (x + L sin th, L cos th), matching the physics
        pivot = (self.sx(x), GROUND_Y - cart_h // 2)
        tip_world = (x + p.pole_length * math.sin(th), p.pole_length * math.cos(th))
        tip = (self.sx(tip_world[0]), self.sy(tip_world[1]))
        upright = abs(th) < 0.2
        pygame.draw.line(s, GOOD if upright else INK, pivot, tip, 7)
        pygame.draw.circle(s, DIM, pivot, 6)
        pygame.draw.circle(s, GOOD if upright else ACCENT, tip, 11)

        self.draw_hud(hud, th, thd, x, xd, action)
        pygame.display.flip()

    def draw_hud(self, hud: dict, th: float, thd: float, x: float, xd: float, action: float) -> None:
        s = self.screen
        mode = hud["mode"]
        mode_colour = {"HUMAN": WARN, "AGENT": ACCENT, "BASELINE": GOOD}[mode]
        s.blit(self.big.render(mode, True, mode_colour), (26, 22))

        if mode == "AGENT":
            e = hud["entry"]
            label = f"checkpoint {hud['index'] + 1}/{hud['n_ckpt']}  |  episode {e['episode']}  |  {e['step']:,} env steps"
            score = (f"trained score {e['eval_return']:.0f}   upright {e['eval_upright_frac']:.0%}"
                     f"   solved {e['eval_solved']:.0%}")
        elif mode == "BASELINE":
            label = "hand-tuned energy pumping + linear balance"
            score = "no learning: needs the equations, the masses and 5 gains"
        else:
            label = "arrow keys push the cart"
            score = "TAB to hand over to the agent"
        s.blit(self.font.render(label, True, INK), (26, 56))
        s.blit(self.small.render(score, True, DIM), (26, 78))

        # --- live episode readout
        rows = [
            ("reward this episode", f"{hud['episode_return']:8.1f}"),
            ("step", f"{hud['steps']:8d} / {self.env.cfg.max_steps}"),
            ("angle from upright", f"{math.degrees(abs(th)):7.1f} deg"),
            ("angular velocity", f"{thd:7.2f} rad/s"),
            ("cart position", f"{x:7.2f} m"),
            ("cart velocity", f"{xd:7.2f} m/s"),
            ("force", f"{action * self.env.p.force_mag:7.2f} N"),
            ("rod energy vs upright", f"{hud['pole_energy']:7.2f} J"),
            ("rod length", f"{self.env.p.pole_length:7.2f} m"),
        ]
        panel = pygame.Rect(WIDTH - 336, 18, 312, 214)
        pygame.draw.rect(s, PANEL, panel, border_radius=8)
        for i, (k, v) in enumerate(rows):
            y = panel.top + 14 + i * 24
            s.blit(self.small.render(k, True, DIM), (panel.left + 14, y))
            s.blit(self.font.render(v, True, INK), (panel.right - 128, y - 2))

        # --- instantaneous reward bar: 1.0/step is the ceiling
        bar = pygame.Rect(26, HEIGHT - 92, 420, 16)
        pygame.draw.rect(s, PANEL, bar, border_radius=8)
        filled = bar.copy()
        filled.width = int(max(0.0, min(1.0, hud["reward"])) * bar.width)
        colour = GOOD if hud["reward"] > 0.75 else (WARN if hud["reward"] > 0.25 else BAD)
        if filled.width > 3:
            pygame.draw.rect(s, colour, filled, border_radius=8)
        s.blit(self.small.render(f"reward now {hud['reward']:.2f} / 1.00", True, DIM), (bar.right + 14, bar.top))

        if hud["paused"]:
            s.blit(self.big.render("PAUSED", True, WARN), (WIDTH // 2 - 58, 22))
        if hud["speed"] != 1.0:
            s.blit(self.font.render(f"speed x{hud['speed']:.2f}", True, WARN), (WIDTH // 2 - 46, 62))
        if hud["last_result"]:
            s.blit(self.font.render(hud["last_result"], True, DIM), (26, HEIGHT - 64))

        keys = "TAB mode   [ ] checkpoint   0-9 jump   E best   R reset   SPACE pause   - = speed   T trail   Q quit"
        s.blit(self.small.render(keys, True, (86, 94, 112)), (26, HEIGHT - 34))


# ------------------------------------------------------------------------ main


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="demo", help="run name under runs/")
    ap.add_argument("--mode", default="human", choices=[m.lower() for m in MODES])
    ap.add_argument("--episode", type=int, default=None, help="start at the checkpoint nearest this episode")
    ap.add_argument("--fps", type=int, default=50, help="render rate; 50 matches the 50 Hz control rate")
    ap.add_argument("--stochastic", action="store_true", help="sample the policy instead of taking its mean")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()

    run_dir = Path("runs") / args.run
    library = CheckpointLibrary(run_dir)
    physics = PhysicsParams(**library.spec["physics"])
    env = CartPoleSwingUpEnv(
        physics=physics,
        config=EnvConfig(**library.spec["config"]),
        reward=RewardParams(**library.spec["reward"]),
    )
    renderer = Renderer(env)
    baseline = EnergyBaseline(env)
    trained_length = physics.pole_length

    mode_index = MODES.index(args.mode.upper())
    # With no --episode given, the best checkpoint is the sensible demo default.
    ckpt_index = library.best() if args.episode is None else library.nearest(args.episode)

    env.reset(seed=0)
    episode_return = 0.0
    reward_now = 0.0
    trail: list[tuple[float, float]] = []
    show_trail = True
    paused = False
    speed = 1.0
    accumulator = 0.0
    last_result = ""
    clock = pygame.time.Clock()
    running = True

    def reset(note: str = "") -> None:
        nonlocal episode_return, reward_now, last_result
        env.reset()
        episode_return = 0.0
        reward_now = 0.0
        trail.clear()
        if note:
            last_result = note

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_TAB:
                    mode_index = (mode_index + 1) % len(MODES)
                    reset()
                elif k == pygame.K_r:
                    reset()
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_LEFTBRACKET:
                    ckpt_index = max(0, ckpt_index - 1)
                    mode_index = MODES.index("AGENT")
                    reset()
                elif k == pygame.K_RIGHTBRACKET:
                    ckpt_index = min(len(library) - 1, ckpt_index + 1)
                    mode_index = MODES.index("AGENT")
                    reset()
                elif k == pygame.K_e:
                    ckpt_index = library.best()
                    mode_index = MODES.index("AGENT")
                    reset()
                elif pygame.K_0 <= k <= pygame.K_9:
                    ckpt_index = library.fraction((k - pygame.K_0) / 10.0)
                    mode_index = MODES.index("AGENT")
                    reset()
                elif k == pygame.K_t:
                    show_trail = not show_trail
                    trail.clear()
                elif k == pygame.K_p:
                    # Impulse straight onto the angular velocity: the cheapest
                    # honest disturbance, since it changes state without
                    # touching the dynamics.
                    kick = 3.5 * (1 if np.random.rand() < 0.5 else -1)
                    env.state[3] += kick
                    last_result = f"poked: {kick:+.1f} rad/s"
                elif k in (pygame.K_COMMA, pygame.K_PERIOD):
                    step_m = 0.1 if k == pygame.K_PERIOD else -0.1
                    new_len = float(np.clip(env.p.pole_length + step_m, 0.4, 2.0))
                    env.p = replace(env.p, pole_length=new_len)
                    physics = env.p  # renderer reads env.p directly
                    last_result = (f"rod now {new_len:.1f} m "
                                   f"(trained on {trained_length:.1f} m)")
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(0.125, speed / 2)
                elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(8.0, speed * 2)

        mode = MODES[mode_index]
        if mode == "HUMAN":
            pressed = pygame.key.get_pressed()
            action = float(pressed[pygame.K_RIGHT]) - float(pressed[pygame.K_LEFT])
        elif mode == "AGENT":
            policy = actor_policy(library.actor(ckpt_index), deterministic=not args.stochastic)
            action = float(policy(env.observe())[0])
        else:
            action = float(baseline(env.observe())[0])

        if not paused:
            # Fractional speeds step the sim on fewer than one frame in one.
            accumulator += speed
            while accumulator >= 1.0:
                accumulator -= 1.0
                _, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                reward_now = reward
                if show_trail:
                    th = wrap_angle(env.state[2])
                    trail.append((env.state[0] + physics.pole_length * math.sin(th),
                                  physics.pole_length * math.cos(th)))
                    del trail[:-160]
                if terminated or truncated:
                    why = "hit the rail end" if info["off_rail"] else "time up"
                    reset(f"last episode: {why}, return {episode_return:.1f}")
                    break

        renderer.draw(
            env.state,
            action,
            {
                "mode": mode,
                "index": ckpt_index,
                "n_ckpt": len(library),
                "entry": library.entries[ckpt_index],
                "episode_return": episode_return,
                "reward": reward_now,
                "steps": env.steps,
                "pole_energy": baseline.pole_energy(),
                "paused": paused,
                "speed": speed,
                "last_result": last_result,
            },
            trail if show_trail else [],
        )
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
