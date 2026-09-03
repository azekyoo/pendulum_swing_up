"""Render the recorded replays as animated GIFs for the README.

    python make_gifs.py

Reads web/replay.json (produced by export_replay.py) rather than re-running any
policy, so this needs only Pillow -- no torch, no checkpoints. Frames are drawn
at 2x and downscaled with Lanczos, because Pillow's draw calls are aliased and
a 1x render of a thin rod looks ragged.

Outputs into docs/:
    agent-balancing.gif   the trained policy, single panel
    learning-stages.gif   episode 0 / 40 / 400 side by side, in lockstep
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Palette shared with the pygame viewer and the slide deck, so the three read
# as one project.
BG = (13, 16, 23)
RAIL = (43, 49, 64)
STOP = (239, 102, 110)
CART = (61, 70, 91)
ROD = (230, 233, 240)
ROD_UP = (95, 211, 154)
TIP = (99, 179, 255)
FORCE = (242, 166, 90)
TRAIL = (44, 74, 112)
GOAL = (34, 64, 47)
LABEL = (168, 177, 196)
LABEL_HI = (230, 233, 240)

SS = 2            # supersampling factor
STRIDE = 3        # keep every Nth simulated step (50 Hz -> ~16.7 fps)
FRAME_MS = 60     # matches STRIDE so playback runs at real speed
MAX_STEPS = 420   # ~8.4 s; the whole swing-up plus a long hold
COLORS = 48       # GIF palette size; the scene is flat, so this is plenty

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap_angle(t: float) -> float:
    return (t + math.pi) % (2 * math.pi) - math.pi


def draw_panel(
    img: ImageDraw.ImageDraw,
    clip: dict,
    step: int,
    physics: dict,
    box: tuple[int, int, int, int],
    label: str | None,
    font,
) -> None:
    """Draw one cart-pole panel inside ``box`` = (left, top, width, height)."""
    left, top, w, h = box
    frames = clip["frames"]
    i = min(step, len(frames) - 1)
    x, _, th, _ = frames[i]
    action = clip["actions"][min(i, len(clip["actions"]) - 1)]

    pole_len, x_limit = physics["pole_length"], physics["x_limit"]
    ground = top + int(h * 0.72)
    scale = min((w - 20 * SS) / (2 * x_limit + 0.7), (ground - top - 16 * SS) / (pole_len * 1.2))
    sx = lambda v: left + w / 2 + v * scale
    sy = lambda v: ground - v * scale

    # panel border and goal height
    img.rectangle([left, top, left + w - 1, top + h - 1], outline=RAIL, width=1 * SS)
    img.line([left + 6 * SS, sy(pole_len), left + w - 6 * SS, sy(pole_len)], fill=GOAL, width=1 * SS)

    # rail and its end stops
    img.line([sx(-x_limit), ground + 10 * SS, sx(x_limit), ground + 10 * SS], fill=RAIL, width=2 * SS)
    for end in (-x_limit, x_limit):
        img.line([sx(end), ground - 3 * SS, sx(end), ground + 17 * SS], fill=STOP, width=2 * SS)

    # tip trail: the widening arcs are the energy going in
    pts = []
    for j in range(max(0, i - 70), i + 1):
        jx, _, jth, _ = frames[j]
        pts.append((sx(jx + pole_len * math.sin(jth)), sy(pole_len * math.cos(jth))))
    if len(pts) > 1:
        img.line(pts, fill=TRAIL, width=1 * SS, joint="curve")

    # applied force
    if abs(action) > 0.02:
        img.line([sx(x), ground + 3 * SS, sx(x) + action * 0.6 * scale, ground + 3 * SS],
                 fill=FORCE, width=3 * SS)

    # cart
    cw, ch = 0.55 * scale, 0.2 * scale
    img.rounded_rectangle([sx(x) - cw / 2, ground - ch / 2, sx(x) + cw / 2, ground + ch / 2],
                          radius=3 * SS, fill=CART)

    # rod
    upright = abs(wrap_angle(th)) < 0.2
    tip = (sx(x + pole_len * math.sin(th)), sy(pole_len * math.cos(th)))
    img.line([sx(x), ground - ch / 2, *tip], fill=ROD_UP if upright else ROD, width=4 * SS)
    r = 5 * SS
    img.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r], fill=ROD_UP if upright else TIP)

    if label:
        img.text((left + 9 * SS, top + 7 * SS), label, font=font, fill=LABEL)
    secs = f"{i * physics['dt']:.1f}s"
    bbox = img.textbbox((0, 0), secs, font=font)
    img.text((left + w - 9 * SS - (bbox[2] - bbox[0]), top + 7 * SS), secs, font=font, fill=LABEL_HI)


def render_gif(out: Path, clips: list[tuple[dict, str | None]], physics: dict,
               panel_w: int, panel_h: int, gap: int = 8) -> None:
    font = load_font(11 * SS)
    n = len(clips)
    W = (panel_w * n + gap * (n - 1)) * SS
    H = panel_h * SS
    longest = max(len(c["frames"]) for c, _ in clips)
    total = min(longest, MAX_STEPS)

    frames: list[Image.Image] = []
    for step in range(0, total, STRIDE):
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        for k, (clip, label) in enumerate(clips):
            x0 = k * (panel_w + gap) * SS
            draw_panel(draw, clip, step, physics, (x0, 0, panel_w * SS, H), label, font)
        canvas = canvas.resize((W // SS, H // SS), Image.LANCZOS)
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=COLORS))

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], optimize=True,
                   duration=FRAME_MS, loop=0, disposal=2)
    kb = out.stat().st_size / 1024
    print(f"{out}  {len(frames)} frames  {kb:.0f} KB")


def main() -> None:
    data = json.loads(Path("web/replay.json").read_text())
    physics = data["spec"]["physics"]
    by_label = {c["label"]: c for c in data["clips"]}

    render_gif(Path("docs/agent-balancing.gif"),
               [(by_label["best"], "trained agent - episode 400")],
               physics, panel_w=560, panel_h=300)

    render_gif(Path("docs/learning-stages.gif"),
               [(by_label["untrained"], "episode 0"),
                (by_label["first time upright"], "episode 40"),
                (by_label["best"], "episode 400")],
               physics, panel_w=272, panel_h=228)


if __name__ == "__main__":
    main()
