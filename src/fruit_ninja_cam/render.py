"""Presentation layer: backdrop treatment, fruit sprites, blade, VFX and HUD.

`Renderer` keeps the frame-to-frame animation state (particles, score easing,
screen shake). `draw_frame` remains as a stateless-looking convenience wrapper
around a module-level renderer.
"""

from __future__ import annotations

import math
import random
import time
from typing import Sequence

import cv2
import numpy as np

from fruit_ninja_cam import config, gfx, theme
from fruit_ninja_cam.effects import Effects
from fruit_ninja_cam.game import Fruit, Game, GameState, SliceEvent
from fruit_ninja_cam.hand_tracker import TrailPoint

_MENU_ICONS = ("Watermelon", "Orange", "Apple", "Lemon", "Grape")


class Renderer:
    """Draws a game frame. One instance per window; holds animation state."""

    def __init__(self) -> None:
        self.effects = Effects()
        self._rng = random.Random(4)
        self._vignette: np.ndarray | None = None
        self._vignette_size: tuple[int, int] = (0, 0)
        self._last_t = time.time()
        self._clock = 0.0
        self._score_shown = 0.0
        self._score_pop = 0.0
        self._prev_score = 0
        self._prev_lives = config.STARTING_LIVES
        self._prev_state: GameState | None = None
        self._state_age = 0.0
        self._last_epoch = -1

    # --- public API ----------------------------------------------------------

    def render(
        self,
        frame_bgr: np.ndarray,
        game: Game,
        trail: Sequence[TrailPoint],
        tip: tuple[float, float] | None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Composite one frame. Pass `dt` to drive animation off a fixed clock."""
        now = time.time()
        if dt is None:
            dt = now - self._last_t
        self._last_t = now
        dt = min(0.05, max(0.0, dt))
        self._clock += dt

        h, w = frame_bgr.shape[:2]
        self._sync_state(game, dt)
        if game.event_epoch != self._last_epoch:
            self._last_epoch = game.event_epoch
            self._consume_events(game.last_events)
        self.effects.update(dt, w, h)

        out = self._backdrop(frame_bgr)
        self.effects.draw_under(out)
        self._draw_fruits(out, game.fruits)
        self._draw_blade(out, trail, tip)
        self.effects.draw_over(out)
        out = self._apply_shake(out)

        if game.state != GameState.MENU:
            self._draw_hud(out, game)
        if game.state == GameState.MENU:
            self._draw_menu(out)
        elif game.state == GameState.GAME_OVER:
            self._draw_game_over(out, game)
        return out

    def reset(self) -> None:
        """Clear transient effects — call when a new run starts."""
        self.effects.clear()
        self._score_shown = 0.0
        self._prev_score = 0
        self._prev_lives = config.STARTING_LIVES

    # --- state tracking ------------------------------------------------------

    def _sync_state(self, game: Game, dt: float) -> None:
        if game.state != self._prev_state:
            self._prev_state = game.state
            self._state_age = 0.0
            if game.state == GameState.PLAYING:
                self._prev_lives = game.lives
                self._prev_score = game.score
                self._score_shown = float(game.score)
        else:
            self._state_age += dt

        if game.score != self._prev_score:
            self._score_pop = 1.0
            self._prev_score = game.score
        self._score_pop *= 0.88 ** (dt * 60.0)
        self._score_shown += (game.score - self._score_shown) * min(1.0, dt * 12.0)

        if game.lives < self._prev_lives and game.state == GameState.PLAYING:
            self.effects.flash(config.HUD_DANGER, 0.38)
        self._prev_lives = game.lives

    def _consume_events(self, events: Sequence[SliceEvent]) -> None:
        for ev in events:
            if ev.is_bomb:
                self.effects.explode(ev.x, ev.y, ev.radius)
                continue
            self.effects.slice_fruit(
                name=ev.name,
                radius=ev.radius,
                skin=ev.color_bgr,
                x=ev.x,
                y=ev.y,
                vx=ev.vx,
                vy=ev.vy,
                angle_deg=ev.angle_deg,
            )
            self.effects.popup(ev.x, ev.y - ev.radius * 0.6, f"+{ev.points}", theme.GOLD, 0.9)
            if ev.combo >= 2:
                self.effects.popup(
                    ev.x, ev.y - ev.radius * 1.35, f"COMBO x{ev.combo}", theme.MINT, 0.7
                )

    # --- backdrop ------------------------------------------------------------

    def _backdrop(self, frame: np.ndarray) -> np.ndarray:
        """Push the webcam feed back so game art reads over any room lighting."""
        h, w = frame.shape[:2]
        out = frame
        if config.BACKDROP_DESATURATE > 0:
            grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(
                frame, 1.0 - config.BACKDROP_DESATURATE, grey, config.BACKDROP_DESATURATE, 0
            )

        if self._vignette is None or self._vignette_size != (h, w):
            vig = gfx.vignette_mask(h, w, config.VIGNETTE_STRENGTH).astype(np.float32)
            vig *= 1.0 - config.BACKDROP_DIM
            self._vignette = vig.astype(np.uint8)
            self._vignette_size = (h, w)
        return cv2.multiply(out, self._vignette, scale=1.0 / 255.0)

    def _apply_shake(self, frame: np.ndarray) -> np.ndarray:
        if not config.SCREEN_SHAKE:
            return frame
        ox, oy = self.effects.shake_offset
        if abs(ox) < 0.4 and abs(oy) < 0.4:
            return frame
        m = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy]], np.float32)
        return cv2.warpAffine(
            frame, m, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE
        )

    # --- fruit ---------------------------------------------------------------

    def _draw_fruits(self, frame: np.ndarray, fruits: Sequence[Fruit]) -> None:
        for f in fruits:
            if not f.alive:
                continue
            if f.is_bomb:
                self._draw_bomb(frame, f)
            else:
                r = int(round(f.radius))
                if config.FRUIT_SHADOW:
                    gfx.blit(frame, _shadow_sprite(r), f.x + r * 0.10, f.y + r * 0.14, 0.85)
                sprite = theme.fruit_sprite(f.name, r, f.color_bgr)
                spin = (f.x + f.y) * 0.35  # cheap per-fruit tumble from its own path
                gfx.blit(frame, gfx.rotated(sprite.bgra, spin % 360.0), f.x, f.y)
                if config.SHOW_FRUIT_LABELS:
                    gfx.text(
                        frame, f.name, int(f.x), int(f.y + r + 20), scale=0.45, center=True
                    )

    def _draw_bomb(self, frame: np.ndarray, bomb: Fruit) -> None:
        r = int(round(bomb.radius))
        pulse = 0.5 + 0.5 * math.sin(self._clock * 7.0)
        gfx.blit(frame, _shadow_sprite(r), bomb.x + r * 0.10, bomb.y + r * 0.14, 0.9)
        gfx.blit_add(
            frame, _glow_sprite(int(r * 1.9), (40, 40, 210)), bomb.x, bomb.y, 0.35 + 0.35 * pulse
        )
        gfx.blit(frame, theme.bomb_sprite(r).bgra, bomb.x, bomb.y)

        # Fuse whips as the bomb flies; the spark is the brightest thing on screen.
        cx, cy = bomb.x - r * 0.10, bomb.y - r * 0.86
        wobble = math.sin(self._clock * 9.0) * r * 0.16
        p0 = (int(cx), int(cy))
        p1 = (int(cx + r * 0.30 + wobble), int(cy - r * 0.38))
        p2 = (int(cx - r * 0.10 + wobble * 1.4), int(cy - r * 0.72))
        cv2.polylines(frame, [np.array([p0, p1, p2], np.int32)], False, (48, 62, 96), max(2, r // 12), cv2.LINE_AA)
        spark_r = int(r * (0.16 + 0.05 * pulse))
        gfx.blit_add(frame, _glow_sprite(int(r * 0.9), (90, 200, 255)), p2[0], p2[1], 0.8 + 0.2 * pulse)
        cv2.circle(frame, p2, max(2, spark_r), (200, 240, 255), -1, cv2.LINE_AA)

        if self._rng.random() < 0.5:
            self.effects.particles.append(
                _spark_at(self._rng, float(p2[0]), float(p2[1]), r)
            )

    # --- blade ---------------------------------------------------------------

    def _draw_blade(
        self,
        frame: np.ndarray,
        trail: Sequence[TrailPoint],
        tip: tuple[float, float] | None,
    ) -> None:
        pts = _smooth([(float(p.x), float(p.y)) for p in trail])
        if len(pts) >= 2:
            _draw_ribbon(frame, pts, config.BLADE_WIDTH, theme.BLADE_CORE, theme.BLADE_GLOW)

        if tip is None:
            return
        tx, ty = int(tip[0]), int(tip[1])
        gfx.blit_add(frame, _glow_sprite(26, (120, 180, 255)), tx, ty, 0.9)
        cv2.circle(frame, (tx, ty), 7, (255, 255, 255), -1, cv2.LINE_AA)
        ring = 15 + int(2.0 * math.sin(self._clock * 6.0))
        cv2.circle(frame, (tx, ty), ring, theme.GOLD, 2, cv2.LINE_AA)
        for i in range(4):
            a = self._clock * 2.4 + i * math.pi / 2
            x0 = int(tx + math.cos(a) * (ring + 4))
            y0 = int(ty + math.sin(a) * (ring + 4))
            x1 = int(tx + math.cos(a) * (ring + 11))
            y1 = int(ty + math.sin(a) * (ring + 11))
            cv2.line(frame, (x0, y0), (x1, y1), theme.GOLD, 2, cv2.LINE_AA)

    # --- HUD -----------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, game: Game) -> None:
        h, w = frame.shape[:2]
        gfx.panel(frame, 18, 14, 262, 96, radius=20, alpha=0.5, border=(90, 84, 96), border_alpha=0.35)
        gfx.text(frame, "SCORE", 40, 48, scale=0.52, color=(172, 172, 182), thickness=1)
        pop = 1.0 + 0.20 * self._score_pop
        gfx.text(
            frame,
            f"{int(round(self._score_shown)):,}",
            40,
            94,
            scale=1.3 * pop,
            color=theme.GOLD if self._score_pop > 0.05 else (245, 245, 250),
            thickness=2,
            font=gfx.FONT_HEAVY,
        )

        lives_w = 56 + config.STARTING_LIVES * 42
        lx = w - lives_w - 18
        gfx.panel(frame, lx, 14, lives_w, 96, radius=20, alpha=0.5, border=(90, 84, 96), border_alpha=0.35)
        gfx.text(frame, "LIVES", lx + 24, 48, scale=0.52, color=(172, 172, 182), thickness=1)
        for i in range(config.STARTING_LIVES):
            filled = i < game.lives
            cx = lx + 38 + i * 42
            wobble = 0
            if filled and game.lives == 1:
                wobble = int(2.5 * math.sin(self._clock * 9.0))
            gfx.heart(
                frame,
                cx,
                80 + wobble,
                13,
                config.HUD_DANGER if filled else (86, 82, 92),
                filled,
            )

        combo = game.combo if game.state == GameState.PLAYING else 0
        if combo >= 2:
            pulse = 1.0 + 0.06 * math.sin(self._clock * 12.0)
            gfx.text(
                frame,
                f"COMBO x{combo}",
                w // 2,
                78,
                scale=0.95 * pulse,
                color=theme.MINT,
                thickness=2,
                center=True,
                font=gfx.FONT_HEAVY,
            )

    # --- overlays ------------------------------------------------------------

    def _dim(self, frame: np.ndarray, amount: float, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        gfx.tint(frame, color, amount)

    def _draw_menu(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._dim(frame, 0.42)
        cy = h // 2

        self._draw_icon_row(frame, cy - 148)
        self._title(frame, "FRUIT NINJA", w // 2, cy - 34, scale=2.0, color=theme.GOLD)
        gfx.text(
            frame,
            "C A M",
            w // 2,
            cy + 6,
            scale=0.85,
            color=(210, 214, 224),
            thickness=2,
            center=True,
        )
        _slash(frame, w // 2, cy + 26, int(w * 0.22), theme.BLADE_GLOW)

        breathe = 0.5 + 0.5 * math.sin(self._clock * 3.2)
        gfx.text(
            frame,
            "Raise your index finger and slash",
            w // 2,
            cy + 76,
            scale=0.72,
            color=(230, 232, 240),
            thickness=1,
            center=True,
        )
        gfx.text(
            frame,
            "PRESS  SPACE  TO PLAY",
            w // 2,
            cy + 124,
            scale=0.8,
            color=tuple(int(120 + 135 * breathe) for _ in range(3)),
            thickness=2,
            center=True,
            font=gfx.FONT_HEAVY,
        )
        self._chips(frame, cy + 168, [("SPACE", "start"), ("Q", "quit"), ("BOMB", "game over")])

    def _draw_game_over(self, frame: np.ndarray, game: Game) -> None:
        h, w = frame.shape[:2]
        # Hold the banner back briefly so the explosion that killed the run
        # actually gets seen before the end card covers it.
        delay = 0.65
        if self._state_age < delay:
            self._dim(frame, 0.30 * (self._state_age / delay), (24, 10, 16))
            return
        intro = min(1.0, (self._state_age - delay) * 3.0)
        self._dim(frame, 0.30 + 0.24 * intro, (24, 10, 16))
        cy = h // 2

        self._title(frame, "GAME OVER", w // 2, cy - 62, scale=1.9, color=(96, 92, 255))
        _slash(frame, w // 2, cy - 24, int(w * 0.20), (72, 68, 255))

        panel_w = 360
        gfx.panel(frame, w // 2 - panel_w // 2, cy + 6, panel_w, 106, radius=20, alpha=0.55, border=(120, 110, 130))
        gfx.text(frame, "FINAL SCORE", w // 2, cy + 44, scale=0.55, color=(178, 178, 190), thickness=1, center=True)
        gfx.text(
            frame,
            f"{game.score:,}",
            w // 2,
            cy + 96,
            scale=1.5,
            color=theme.GOLD,
            thickness=2,
            center=True,
            font=gfx.FONT_HEAVY,
        )
        breathe = 0.5 + 0.5 * math.sin(self._clock * 3.2)
        gfx.text(
            frame,
            "SPACE to slice again    Q to quit",
            w // 2,
            cy + 160,
            scale=0.7,
            color=tuple(int(130 + 125 * breathe) for _ in range(3)),
            thickness=1,
            center=True,
        )

    def _title(self, frame: np.ndarray, label: str, cx: int, y: int, scale: float, color) -> None:
        (tw, th), _ = cv2.getTextSize(label, gfx.FONT_HEAVY, scale, 3)
        x = cx - tw // 2
        pad = 34
        glow = np.zeros((th + 2 * pad, tw + 2 * pad, 3), np.uint8)
        cv2.putText(glow, label, (pad, pad + th), gfx.FONT_HEAVY, scale, color, 7, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 13.0)
        gfx.blit_add(frame, glow, cx, y - (pad + th) + glow.shape[0] / 2.0, 0.9)
        gfx.text(
            frame, label, x, y, scale=scale, color=color, thickness=3, font=gfx.FONT_HEAVY, outline=(12, 10, 16)
        )

    def _draw_icon_row(self, frame: np.ndarray, y: int) -> None:
        w = frame.shape[1]
        step = 118
        x0 = w // 2 - (len(_MENU_ICONS) - 1) * step // 2
        for i, name in enumerate(_MENU_ICONS):
            bob = math.sin(self._clock * 2.2 + i * 0.8) * 9.0
            sprite = theme.fruit_sprite(name, 38)
            gfx.blit(frame, gfx.rotated(sprite.bgra, self._clock * 22.0 + i * 40.0), x0 + i * step, y + bob)

    def _chips(self, frame: np.ndarray, y: int, items: list[tuple[str, str]]) -> None:
        w = frame.shape[1]
        widths = [gfx.text_width(k, 0.55, 2) + gfx.text_width(v, 0.55, 1) + 54 for k, v in items]
        total = sum(widths) + 18 * (len(items) - 1)
        x = w // 2 - total // 2
        for (key, label), cw in zip(items, widths):
            gfx.panel(frame, x, y, cw, 40, radius=14, alpha=0.5, border=(110, 104, 118), border_alpha=0.4)
            gfx.text(frame, key, x + 16, y + 27, scale=0.55, color=theme.GOLD, thickness=2, shadow=False)
            gfx.text(
                frame,
                label,
                x + 30 + gfx.text_width(key, 0.55, 2),
                y + 27,
                scale=0.55,
                color=(206, 208, 216),
                thickness=1,
                shadow=False,
            )
            x += cw + 18


# --- sprite helpers ----------------------------------------------------------

_SHADOWS: dict[int, np.ndarray] = {}
_GLOWS: dict[tuple[int, tuple[int, int, int]], np.ndarray] = {}


def _shadow_sprite(radius: int) -> np.ndarray:
    """Soft dark halo that separates a fruit from a busy webcam backdrop."""
    cached = _SHADOWS.get(radius)
    if cached is not None:
        return cached
    size = int(radius * 2.9)
    sprite = np.zeros((size, size, 4), np.uint8)
    cv2.circle(sprite, (size // 2, size // 2), int(radius * 1.02), (6, 4, 10, 190), -1, cv2.LINE_AA)
    sprite = cv2.GaussianBlur(sprite, (0, 0), radius * 0.30)
    _SHADOWS[radius] = sprite
    return sprite


def _glow_sprite(radius: int, color: tuple[int, int, int]) -> np.ndarray:
    key = (radius, color)
    cached = _GLOWS.get(key)
    if cached is None:
        cached = gfx.radial_glow(radius, color)
        _GLOWS[key] = cached
    return cached


def _spark_at(rng: random.Random, x: float, y: float, r: int):
    from fruit_ninja_cam.effects import Particle

    a = rng.uniform(0, 2 * math.pi)
    return Particle(
        x=x,
        y=y,
        vx=math.cos(a) * rng.uniform(20.0, 120.0),
        vy=math.sin(a) * rng.uniform(20.0, 120.0) - 40.0,
        radius=rng.uniform(1.5, max(2.0, r * 0.08)),
        color=(90, 200, 255),
        ttl=rng.uniform(0.15, 0.35),
        gravity=180.0,
        additive=True,
    )


def _smooth(pts: Sequence[tuple[float, float]], passes: int = 2) -> list[tuple[float, float]]:
    """Chaikin-style smoothing so tracker jitter doesn't kink the blade."""
    out = list(pts)
    for _ in range(passes):
        if len(out) < 3:
            break
        smoothed = [out[0]]
        for i in range(1, len(out) - 1):
            px, py = out[i - 1]
            cx, cy = out[i]
            nx, ny = out[i + 1]
            smoothed.append(((px + 2 * cx + nx) / 4.0, (py + 2 * cy + ny) / 4.0))
        smoothed.append(out[-1])
        out = smoothed
    return out


def _ribbon_edges(
    pts: Sequence[tuple[float, float]], width: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Offset a polyline into two edges, tapering from a point at the tail."""
    n = len(pts)
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(pts):
        if i == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif i == n - 1:
            dx, dy = x - pts[-2][0], y - pts[-2][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            dx, dy, length = 1.0, 0.0, 1.0
        nx, ny = -dy / length, dx / length
        t = i / (n - 1)
        half = max(0.4, width * (t**0.8) * 0.5)
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    return left, right


def _draw_ribbon(
    frame: np.ndarray,
    pts: Sequence[tuple[float, float]],
    width: float,
    core: tuple[int, int, int],
    glow: tuple[int, int, int] | None = None,
) -> None:
    """Blade swipe: a tapered ribbon that also fades out toward its tail."""
    n = len(pts)
    if n < 2:
        return
    left, right = _ribbon_edges(pts, width)
    xs = [p[0] for p in left + right]
    ys = [p[1] for p in left + right]
    pad = 24
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(frame.shape[1], int(max(xs)) + pad)
    y1 = min(frame.shape[0], int(max(ys)) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return

    stencil = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], np.float32)
    for i in range(1, n):
        quad = np.array(
            [left[i - 1], left[i], right[i], right[i - 1]], np.float32
        ) - off
        # Oldest segments are faintest; newest (nearest the fingertip) is solid.
        alpha = int(255 * (i / (n - 1)) ** 0.7)
        cv2.fillPoly(stencil, [quad.astype(np.int32)], alpha, cv2.LINE_AA)
    stencil = cv2.GaussianBlur(stencil, (0, 0), 0.8)

    roi = frame[y0:y1, x0:x1]
    if glow is not None and config.BLADE_GLOW:
        halo = cv2.GaussianBlur(stencil, (0, 0), 10.0)
        cv2.add(roi, (halo[..., None].astype(np.float32) / 255.0 * np.array(glow, np.float32)).astype(np.uint8), roi)
    gfx.blend_mask(roi, stencil, core)


def _slash(frame: np.ndarray, cx: int, y: int, half_len: int, color: tuple[int, int, int]) -> None:
    """Decorative blade swipe used as a divider on the menu screens."""
    pts = [
        (float(cx - half_len), float(y + 7)),
        (float(cx), float(y)),
        (float(cx + half_len), float(y - 7)),
    ]
    _draw_ribbon(frame, pts, 13.0, color, tuple(int(c * 0.5) for c in color))


# --- module-level convenience ------------------------------------------------

_default_renderer: Renderer | None = None


def get_renderer() -> Renderer:
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = Renderer()
    return _default_renderer


def draw_frame(
    frame_bgr: np.ndarray,
    game: Game,
    trail: Sequence[TrailPoint],
    tip: tuple[float, float] | None,
) -> np.ndarray:
    """Return a BGR frame with game overlays. Does not mutate input."""
    return get_renderer().render(frame_bgr, game, trail, tip)
