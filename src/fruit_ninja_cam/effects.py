"""Transient visual effects: fruit halves, juice, sparks, shockwaves, shake.

Purely cosmetic and self-contained — the game rules never read this state, so
effects can be tuned (or disabled) without touching gameplay.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from fruit_ninja_cam import gfx, theme

BGR = tuple[int, int, int]

GRAVITY = 1500.0
MAX_PARTICLES = 260


@lru_cache(maxsize=192)
def _glow(radius: int, color: BGR) -> np.ndarray:
    return gfx.radial_glow(radius, color)


@dataclass
class Half:
    """One flying piece of a sliced fruit."""

    bgra: np.ndarray
    x: float
    y: float
    vx: float
    vy: float
    spin: float
    angle: float = 0.0
    age: float = 0.0
    ttl: float = 1.15

    def update(self, dt: float) -> None:
        self.age += dt
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt

    def draw(self, frame: np.ndarray) -> None:
        fade = 1.0 - _ease_in(self.age / self.ttl, 0.65)
        gfx.blit(frame, gfx.rotated(self.bgra, self.angle), self.x, self.y, fade)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: BGR
    ttl: float
    age: float = 0.0
    drag: float = 0.86
    gravity: float = GRAVITY
    additive: bool = False
    shrink: float = 1.0

    def update(self, dt: float) -> None:
        self.age += dt
        self.vy += self.gravity * dt
        damp = self.drag**dt
        self.vx *= damp
        self.vy *= damp
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, frame: np.ndarray) -> None:
        t = min(1.0, self.age / self.ttl)
        fade = 1.0 - t * t
        r = max(1, int(self.radius * (1.0 - t * (1.0 - self.shrink))))
        if self.additive:
            gfx.blit_add(frame, _glow(r * 2, self.color), self.x, self.y, fade)
        else:
            color = tuple(int(c * (0.45 + 0.55 * fade)) for c in self.color)
            cv2.circle(frame, (int(self.x), int(self.y)), r, color, -1, cv2.LINE_AA)


@dataclass
class Popup:
    x: float
    y: float
    label: str
    color: BGR
    ttl: float = 0.85
    age: float = 0.0
    scale: float = 0.85

    def update(self, dt: float) -> None:
        self.age += dt
        self.y -= 62.0 * dt

    def draw(self, frame: np.ndarray) -> None:
        t = min(1.0, self.age / self.ttl)
        pop = 1.0 + 0.35 * math.exp(-9.0 * self.age)
        alpha = 1.0 - t * t
        color = tuple(int(20 + (c - 20) * alpha) for c in self.color)
        gfx.text(
            frame,
            self.label,
            int(self.x),
            int(self.y),
            scale=self.scale * pop,
            color=color,
            thickness=2,
            center=True,
        )


@dataclass
class Shockwave:
    x: float
    y: float
    max_radius: float
    color: BGR
    ttl: float = 0.55
    age: float = 0.0
    thickness: float = 10.0

    def update(self, dt: float) -> None:
        self.age += dt

    def draw(self, frame: np.ndarray) -> None:
        t = min(1.0, self.age / self.ttl)
        r = int(self.max_radius * _ease_out(t))
        if r < 2:
            return
        fade = (1.0 - t) ** 1.6
        thick = max(1, int(self.thickness * (1 - t) + 1))
        pad = thick + 6
        x0, y0 = int(self.x) - r - pad, int(self.y) - r - pad
        span = 2 * (r + pad)
        layer = np.zeros((span, span, 3), np.uint8)
        cv2.circle(
            layer,
            (span // 2, span // 2),
            r,
            tuple(int(c * fade) for c in self.color),
            thick,
            cv2.LINE_AA,
        )
        layer = cv2.GaussianBlur(layer, (0, 0), 2.0 + thick * 0.35)
        # Additive: a shockwave is light, not paint.
        gfx.blit_add(frame, layer, x0 + span / 2.0, y0 + span / 2.0)


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _ease_in(t: float, start: float) -> float:
    """0 until `start` of the lifetime, then ramps to 1 — a late fade-out."""
    if t <= start:
        return 0.0
    return min(1.0, (t - start) / max(1e-3, 1.0 - start))


class Effects:
    """Owns every transient particle/sprite spawned by gameplay events."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.halves: list[Half] = []
        self.particles: list[Particle] = []
        self.popups: list[Popup] = []
        self.waves: list[Shockwave] = []
        self._shake = 0.0
        self._flash = 0.0
        self._flash_color: BGR = (255, 255, 255)
        self._t = 0.0

    # --- spawning ------------------------------------------------------------

    def slice_fruit(
        self,
        *,
        name: str,
        radius: float,
        skin: BGR,
        x: float,
        y: float,
        vx: float,
        vy: float,
        angle_deg: float,
    ) -> None:
        """Split a fruit into two flying halves plus a burst of juice."""
        r = int(max(8, round(radius)))
        style = theme.style_for(name, skin)
        sprite = theme.fruit_sprite(name, r, skin)
        face = theme.cut_face(name, r, skin)
        top, bottom = _bake_halves(sprite.bgra, face, angle_deg, style.skin_shadow)

        push = self._rng.uniform(150.0, 250.0)
        nx = -math.sin(math.radians(angle_deg))
        ny = math.cos(math.radians(angle_deg))
        for piece, sign in ((top, -1.0), (bottom, 1.0)):
            self.halves.append(
                Half(
                    bgra=piece,
                    x=x + nx * sign * r * 0.12,
                    y=y + ny * sign * r * 0.12,
                    vx=vx * 0.55 + nx * sign * push,
                    vy=vy * 0.55 + ny * sign * push - 60.0,
                    spin=self._rng.uniform(90.0, 260.0) * sign,
                )
            )

        self._spawn_juice(x, y, r, style.juice, angle_deg)
        self._shake = max(self._shake, 3.0)

    def _spawn_juice(self, x: float, y: float, r: int, color: BGR, angle_deg: float) -> None:
        rng = self._rng
        for _ in range(30):
            a = math.radians(angle_deg + rng.uniform(-38.0, 38.0) + (180.0 if rng.random() < 0.5 else 0.0))
            speed = rng.uniform(180.0, 700.0)
            self._add(
                Particle(
                    x=x + rng.uniform(-r * 0.3, r * 0.3),
                    y=y + rng.uniform(-r * 0.3, r * 0.3),
                    vx=math.cos(a) * speed,
                    vy=math.sin(a) * speed - rng.uniform(0.0, 160.0),
                    radius=rng.uniform(3.0, r * 0.26),
                    color=color,
                    ttl=rng.uniform(0.45, 0.9),
                    shrink=0.3,
                )
            )
        for _ in range(8):
            a = math.radians(rng.uniform(0.0, 360.0))
            self._add(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(a) * rng.uniform(40.0, 190.0),
                    vy=math.sin(a) * rng.uniform(40.0, 190.0),
                    radius=rng.uniform(r * 0.12, r * 0.26),
                    color=_lighten(color, 0.45),
                    ttl=rng.uniform(0.25, 0.45),
                    gravity=260.0,
                    additive=True,
                )
            )
        # A brief bloom at the cut point instead of a ring — a ring reads as a
        # bubble at fruit scale.
        self._add(
            Particle(
                x=x,
                y=y,
                vx=0.0,
                vy=0.0,
                radius=r * 0.85,
                color=_lighten(color, 0.5),
                ttl=0.14,
                gravity=0.0,
                additive=True,
                shrink=0.4,
            )
        )

    def explode(self, x: float, y: float, radius: float) -> None:
        """Bomb detonation: sparks, smoke, shockwaves, flash and a hard shake."""
        rng = self._rng
        r = max(10.0, radius)
        self.waves.append(Shockwave(x=x, y=y, max_radius=r * 7.5, color=(140, 200, 255), ttl=0.5, thickness=14.0))
        self.waves.append(Shockwave(x=x, y=y, max_radius=r * 4.5, color=(80, 140, 255), ttl=0.35, thickness=22.0))
        for _ in range(70):
            a = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(200.0, 1250.0)
            self._add(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(a) * speed,
                    vy=math.sin(a) * speed,
                    radius=rng.uniform(2.0, 7.0),
                    color=rng.choice([(60, 190, 255), (40, 120, 255), (170, 230, 255)]),
                    ttl=rng.uniform(0.35, 0.8),
                    gravity=520.0,
                    additive=True,
                )
            )
        for _ in range(22):
            a = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(30.0, 260.0)
            self._add(
                Particle(
                    x=x + rng.uniform(-r, r),
                    y=y + rng.uniform(-r, r),
                    vx=math.cos(a) * speed,
                    vy=math.sin(a) * speed - 60.0,
                    radius=rng.uniform(r * 0.35, r * 0.85),
                    color=(52, 48, 56),
                    ttl=rng.uniform(0.6, 1.1),
                    gravity=-120.0,
                    shrink=1.9,
                )
            )
        self._shake = 26.0
        self.flash((120, 190, 255), 0.85)

    def popup(self, x: float, y: float, label: str, color: BGR, scale: float = 0.85) -> None:
        self.popups.append(Popup(x=x, y=y, label=label, color=color, scale=scale))

    def flash(self, color: BGR, strength: float) -> None:
        self._flash = max(self._flash, strength)
        self._flash_color = color

    def _add(self, particle: Particle) -> None:
        if len(self.particles) < MAX_PARTICLES:
            self.particles.append(particle)

    # --- simulation ----------------------------------------------------------

    def update(self, dt: float, width: int, height: int) -> None:
        dt = max(0.0, min(0.05, dt))
        self._t += dt
        margin = 220

        for h in self.halves:
            h.update(dt)
        self.halves = [
            h for h in self.halves if h.age < h.ttl and h.y < height + margin and -margin < h.x < width + margin
        ]

        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.age < p.ttl and p.y < height + margin]

        for u in self.popups:
            u.update(dt)
        self.popups = [u for u in self.popups if u.age < u.ttl]

        for w in self.waves:
            w.update(dt)
        self.waves = [w for w in self.waves if w.age < w.ttl]

        self._shake *= 0.86**(dt * 60.0)
        if self._shake < 0.4:
            self._shake = 0.0
        self._flash *= 0.82**(dt * 60.0)
        if self._flash < 0.01:
            self._flash = 0.0

    # --- drawing -------------------------------------------------------------

    def draw_under(self, frame: np.ndarray) -> None:
        """Layers that belong behind live fruit and the blade."""
        for w in self.waves:
            w.draw(frame)
        for h in self.halves:
            h.draw(frame)
        for p in self.particles:
            if not p.additive:
                p.draw(frame)

    def draw_over(self, frame: np.ndarray) -> None:
        """Sparks, popups and the full-frame flash sit on top of everything."""
        for p in self.particles:
            if p.additive:
                p.draw(frame)
        for u in self.popups:
            u.draw(frame)
        if self._flash > 0.01:
            gfx.tint(frame, self._flash_color, self._flash * 0.55)

    @property
    def shake_offset(self) -> tuple[float, float]:
        if self._shake <= 0.0:
            return (0.0, 0.0)
        a = self._t * 46.0
        return (
            math.sin(a * 1.7) * self._shake,
            math.cos(a * 2.3) * self._shake * 0.72,
        )

    def clear(self) -> None:
        self.halves.clear()
        self.particles.clear()
        self.popups.clear()
        self.waves.clear()
        self._shake = 0.0
        self._flash = 0.0


# --- half baking -------------------------------------------------------------


def _lighten(color: BGR, t: float) -> BGR:
    return tuple(int(c + (255 - c) * t) for c in color)  # type: ignore[return-value]


def _bake_halves(
    sprite: np.ndarray, face: np.ndarray, angle_deg: float, edge_color: BGR
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fruit sprite along `angle_deg`, exposing a foreshortened inner face."""
    size = sprite.shape[0]
    cut = gfx.rotated(sprite, -angle_deg)
    cy = size // 2

    face_w = min(size, int(face.shape[1] * 0.96))
    face_h = max(4, int(face_w * 0.26))
    squashed = cv2.resize(face, (face_w, face_h), interpolation=cv2.INTER_AREA)
    face_layer = np.zeros_like(cut)
    x0 = (size - face_w) // 2
    y0 = cy - face_h // 2
    face_layer[y0 : y0 + face_h, x0 : x0 + face_w] = squashed

    halves: list[np.ndarray] = []
    for is_top in (True, False):
        piece = cut.copy()
        if is_top:
            piece[cy:, :, 3] = 0
        else:
            piece[:cy, :, 3] = 0

        lip = face_layer.copy()
        if is_top:
            lip[cy:, :, 3] = 0
        else:
            lip[:cy, :, 3] = 0
        # The exposed face only shows where the fruit body actually is.
        lip[:, :, 3] = (lip[:, :, 3].astype(np.float32) * (piece[:, :, 3].astype(np.float32) / 255.0)).astype(
            np.uint8
        )
        a = lip[:, :, 3:4].astype(np.float32) / 255.0
        piece[:, :, :3] = (
            lip[:, :, :3].astype(np.float32) * a + piece[:, :, :3].astype(np.float32) * (1.0 - a)
        ).astype(np.uint8)

        # A crisp dark line right on the cut sells the separation.
        row = cy - 1 if is_top else cy
        band = piece[max(0, row - 1) : row + 1]
        alpha = band[:, :, 3:4].astype(np.float32) / 255.0
        band[:, :, :3] = (
            np.array(edge_color, np.float32) * alpha + band[:, :, :3].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

        halves.append(gfx.rotated(piece, angle_deg))
    return halves[0], halves[1]
