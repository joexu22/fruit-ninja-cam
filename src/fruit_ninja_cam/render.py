"""Draw mirrored-frame overlays: fruit, bombs, blade trail, HUD."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from fruit_ninja_cam import config
from fruit_ninja_cam.game import Fruit, Game, GameState, SliceEvent
from fruit_ninja_cam.hand_tracker import TrailPoint


def draw_frame(
    frame_bgr: np.ndarray,
    game: Game,
    trail: Sequence[TrailPoint],
    tip: tuple[float, float] | None,
) -> np.ndarray:
    """Return a BGR frame with game overlays. Does not mutate input."""
    out = frame_bgr.copy()
    _draw_fruits(out, game.fruits)
    _draw_trail(out, trail, tip)
    _draw_slice_popups(out, game.last_events)
    _draw_hud(out, game)
    if game.state == GameState.MENU:
        _draw_center_banner(
            out,
            "FRUIT NINJA CAM",
            "Point index finger at camera  |  SPACE to start  |  Q quit",
        )
    elif game.state == GameState.GAME_OVER:
        _draw_center_banner(
            out,
            "GAME OVER",
            f"Score: {game.score}  |  SPACE to restart  |  Q quit",
        )
    return out


def _draw_fruits(frame: np.ndarray, fruits: Sequence[Fruit]) -> None:
    for f in fruits:
        if not f.alive:
            continue
        center = (int(f.x), int(f.y))
        r = int(f.radius)
        if f.is_bomb:
            cv2.circle(frame, center, r, (30, 30, 30), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, center, r, (0, 0, 255), 3, lineType=cv2.LINE_AA)
            # Fuse
            fuse_end = (center[0], center[1] - r - 12)
            cv2.line(frame, (center[0], center[1] - r + 4), fuse_end, (200, 200, 200), 2)
            cv2.circle(frame, fuse_end, 5, (0, 200, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                "BOMB",
                (center[0] - 28, center[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(frame, center, r, f.color_bgr, -1, lineType=cv2.LINE_AA)
            # Highlight
            hi = (
                min(255, f.color_bgr[0] + 60),
                min(255, f.color_bgr[1] + 60),
                min(255, f.color_bgr[2] + 60),
            )
            cv2.circle(
                frame,
                (center[0] - r // 3, center[1] - r // 3),
                max(4, r // 4),
                hi,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(frame, center, r, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                f.name,
                (center[0] - 30, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )


def _draw_trail(
    frame: np.ndarray,
    trail: Sequence[TrailPoint],
    tip: tuple[float, float] | None,
) -> None:
    if len(trail) >= 2:
        pts = np.array([[int(p.x), int(p.y)] for p in trail], dtype=np.int32)
        # Fade older segments by drawing thicker → thinner toward tip
        for i in range(1, len(pts)):
            alpha = i / (len(pts) - 1)
            thickness = max(2, int(2 + 10 * alpha))
            color = (
                int(80 + 175 * alpha),  # B
                int(80 + 100 * alpha),  # G
                255,                    # R — cyan/white blade
            )
            cv2.line(
                frame,
                tuple(pts[i - 1]),
                tuple(pts[i]),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
    if tip is not None:
        cv2.circle(
            frame,
            (int(tip[0]), int(tip[1])),
            8,
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            (int(tip[0]), int(tip[1])),
            12,
            (255, 200, 80),
            2,
            lineType=cv2.LINE_AA,
        )


def _draw_slice_popups(frame: np.ndarray, events: Sequence[SliceEvent]) -> None:
    for ev in events:
        cv2.putText(
            frame,
            f"+{ev.points}",
            (int(ev.x) - 20, int(ev.y) - int(40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_hud(frame: np.ndarray, game: Game) -> None:
    h, w = frame.shape[:2]
    # Translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(
        frame,
        f"Score: {game.score}",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    lives_str = " ".join("♥" if i < game.lives else "♡" for i in range(config.STARTING_LIVES))
    # OpenCV Hershey fonts lack hearts — use ASCII fallback that's always visible
    lives_str = "Lives: " + " ".join(
        "*" if i < game.lives else "-" for i in range(config.STARTING_LIVES)
    )
    cv2.putText(
        frame,
        lives_str,
        (w - 220, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (80, 80, 255) if game.lives <= 1 else (180, 220, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_center_banner(frame: np.ndarray, title: str, subtitle: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (w // 8, h // 2 - 70),
        (7 * w // 8, h // 2 + 70),
        (15, 15, 15),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(
        frame,
        title,
        ((w - tw) // 2, h // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 220, 255),
        3,
        cv2.LINE_AA,
    )
    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(
        frame,
        subtitle,
        ((w - sw) // 2, h // 2 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
