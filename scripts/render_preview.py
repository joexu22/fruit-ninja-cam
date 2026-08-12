#!/usr/bin/env python3
"""Render the game's visuals without a webcam.

Drives the real `Game` + `Renderer` with a scripted fingertip path over a
synthetic backdrop, so the art can be iterated on (and screenshotted) headless.

    python scripts/render_preview.py                # stills into preview/
    python scripts/render_preview.py --video        # also writes preview/demo.mp4
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruit_ninja_cam import config, gfx, theme  # noqa: E402
from fruit_ninja_cam.effects import Effects  # noqa: E402
from fruit_ninja_cam.game import Game, GameState  # noqa: E402
from fruit_ninja_cam.hand_tracker import TrailPoint  # noqa: E402
from fruit_ninja_cam.render import Renderer  # noqa: E402

FPS = 30
DT = 1.0 / FPS


def make_backdrop(w: int, h: int) -> np.ndarray:
    """A stand-in webcam frame: lit room + a person-ish silhouette."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 96 + 40 * (yy / h)
    base[..., 1] = 84 + 26 * (xx / w)
    base[..., 2] = 78 + 18 * (1.0 - yy / h)

    for cx, cy, r, tint in (
        (w * 0.22, h * 0.18, w * 0.34, (55.0, 40.0, 20.0)),
        (w * 0.82, h * 0.72, w * 0.40, (10.0, 26.0, 48.0)),
    ):
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        falloff = np.clip(1.0 - d, 0.0, 1.0) ** 2
        base += falloff[..., None] * np.array(tint, np.float32)

    frame = np.clip(base, 0, 255).astype(np.uint8)
    person = np.zeros((h, w), np.uint8)
    cv2.ellipse(person, (w // 2, int(h * 1.05)), (int(w * 0.26), int(h * 0.52)), 0, 0, 360, 255, -1)
    cv2.circle(person, (w // 2, int(h * 0.44)), int(h * 0.13), 255, -1)
    person = cv2.GaussianBlur(person, (0, 0), 9.0)
    mask = (person.astype(np.float32) / 255.0)[..., None]
    frame = (frame.astype(np.float32) * (1 - mask * 0.55)).astype(np.uint8)

    rng = np.random.default_rng(3)
    noise = rng.normal(0, 6.0, (h, w, 3)).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class Hand:
    """Scripted fingertip: idles in a figure-eight, lunges at whatever it targets."""

    SPEED = 1500.0

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.points: list[TrailPoint] = []
        self.t = 0.0
        self.x = w * 0.5
        self.y = h * 0.6

    def step(self, dt: float, target: tuple[float, float] | None) -> tuple[float, float]:
        self.t += dt
        if target is None:
            phase = self.t * 1.6
            tx = self.w * (0.5 + 0.30 * math.sin(phase))
            ty = self.h * (0.62 + 0.14 * math.sin(phase * 2.1 + 0.7))
        else:
            # Aim past the fruit so the blade sweeps through, not up to, it.
            tx, ty = target
            tx += (tx - self.x) * 0.35
            ty += (ty - self.y) * 0.35

        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        step = self.SPEED * dt
        if dist > step:
            dx, dy = dx / dist * step, dy / dist * step
        self.x += dx
        self.y += dy

        self.points.append(TrailPoint(x=self.x, y=self.y, t=self.t))
        cutoff = self.t - config.TRAIL_MAX_AGE_SEC
        self.points = [p for p in self.points if p.t >= cutoff][-config.TRAIL_MAX_POINTS :]
        return self.x, self.y


def pick_target(game: Game, want_bomb: bool) -> tuple[float, float] | None:
    """Nearest live fruit of the requested kind, if any is on screen."""
    candidates = [f for f in game.fruits if f.alive and f.is_bomb == want_bomb]
    if not candidates:
        return None
    fruit = min(candidates, key=lambda f: f.y)
    return (fruit.x, fruit.y)


def simulate(out_dir: Path, seconds: float, video: bool, size: tuple[int, int]) -> None:
    w, h = size
    backdrop = make_backdrop(w, h)
    game = Game(width=w, height=h)
    renderer = Renderer()
    hand = Hand(w, h)
    random.seed(17)

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if video:
        writer = cv2.VideoWriter(
            str(out_dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
        )

    total = int(seconds * FPS)
    menu_frames = int(FPS * 1.6)
    finale_at = total - int(FPS * 2.2)  # go bomb-hunting so the demo ends with a bang
    now = 1000.0  # simulated clock; the game accepts any monotonic timebase
    pending: list[tuple[int, str]] = [(menu_frames // 2, "menu")]
    saved: set[str] = set()

    for i in range(total):
        now += DT
        if i == menu_frames:
            game.start(now)

        hunting_bomb = i >= finale_at
        target = None
        if game.state == GameState.PLAYING:
            target = pick_target(game, want_bomb=hunting_bomb)
        tip = hand.step(DT, target)

        game.update(DT, hand.points, now=now)
        events = list(game.last_events)
        frame = renderer.render(backdrop, game, hand.points, tip, dt=DT)

        for ev in events:
            if ev.is_bomb and "bomb" not in saved:
                pending.append((i + 5, "bomb"))
            elif not ev.is_bomb and "slice" not in saved:
                pending.append((i + 6, "slice"))
            elif not ev.is_bomb and "combo" not in saved and ev.combo >= 2:
                pending.append((i + 4, "combo"))
        if game.state == GameState.GAME_OVER and "game_over" not in saved:
            pending.append((i + int(FPS * 1.1), "game_over"))
        if game.state == GameState.PLAYING and i > menu_frames + FPS and "play" not in saved:
            pending.append((i + 1, "play"))

        for at, name in list(pending):
            if i >= at and name not in saved:
                cv2.imwrite(str(out_dir / f"{name}.png"), frame)
                saved.add(name)
                pending.remove((at, name))

        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"wrote {sorted(saved)} to {out_dir}")


def art_sheet(out_dir: Path) -> None:
    """Every fruit whole, sliced and exploded — a contact sheet for art review."""
    names = [name for name, _ in config.FRUIT_TYPES]
    cell = 200
    w, h = cell * (len(names) + 1), cell * 2 + 60
    sheet = np.full((h, w, 3), 26, np.uint8)
    for x in range(0, w, 40):
        cv2.line(sheet, (x, 0), (x, h), (34, 32, 38), 1)

    fx = Effects(seed=5)
    for i, name in enumerate(names):
        cx = i * cell + cell // 2
        sprite = theme.fruit_sprite(name, 52)
        gfx.blit(sheet, sprite.bgra, cx, cell // 2 + 20)
        gfx.text(sheet, name, cx, cell - 12, scale=0.5, color=(200, 200, 210), thickness=1, center=True)
        fx.slice_fruit(
            name=name,
            radius=52,
            skin=dict(config.FRUIT_TYPES)[name],
            x=cx,
            y=cell + cell // 2 - 20,
            vx=0.0,
            vy=0.0,
            angle_deg=18.0,
        )

    for _ in range(8):
        fx.update(0.025, w, h)
    fx.draw_under(sheet)
    fx.draw_over(sheet)

    bx = len(names) * cell + cell // 2
    gfx.blit(sheet, theme.bomb_sprite(52).bgra, bx, cell // 2 + 20)
    gfx.text(sheet, "Bomb", bx, cell - 12, scale=0.5, color=(200, 200, 210), thickness=1, center=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "art_sheet.png"), sheet)
    print(f"wrote art sheet to {out_dir / 'art_sheet.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("preview"))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--video", action="store_true", help="also write demo.mp4")
    ap.add_argument("--sheet", action="store_true", help="write an art contact sheet")
    ap.add_argument("--width", type=int, default=config.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=config.CAMERA_HEIGHT)
    args = ap.parse_args()
    if args.sheet:
        art_sheet(args.out)
    simulate(args.out, args.seconds, args.video, (args.width, args.height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
