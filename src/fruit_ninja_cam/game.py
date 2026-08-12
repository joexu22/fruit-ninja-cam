"""Fruit Ninja game state: spawn, gravity, slicing, lives, score."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

from fruit_ninja_cam import config
from fruit_ninja_cam.hand_tracker import TrailPoint


class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    GAME_OVER = auto()


@dataclass
class Fruit:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    name: str
    color_bgr: tuple[int, int, int]
    alive: bool = True
    is_bomb: bool = False


@dataclass
class SliceEvent:
    """Record of a successful slice this frame (for VFX / scoring feedback)."""

    x: float
    y: float
    name: str
    points: int


@dataclass
class Game:
    """Pure game logic — no OpenCV / MediaPipe dependencies."""

    state: GameState = GameState.MENU
    score: int = 0
    lives: int = config.STARTING_LIVES
    fruits: list[Fruit] = field(default_factory=list)
    spawn_interval: float = config.SPAWN_INTERVAL_START
    _next_spawn_at: float = 0.0
    _last_slice_at: float = 0.0
    _combo: int = 0
    width: int = config.CAMERA_WIDTH
    height: int = config.CAMERA_HEIGHT
    last_events: list[SliceEvent] = field(default_factory=list)

    def start(self) -> None:
        """Begin a new run from menu or game-over."""
        self.state = GameState.PLAYING
        self.score = 0
        self.lives = config.STARTING_LIVES
        self.fruits.clear()
        self.spawn_interval = config.SPAWN_INTERVAL_START
        self._next_spawn_at = time.time() + 0.4
        self._last_slice_at = 0.0
        self._combo = 0
        self.last_events.clear()

    def update(
        self,
        dt: float,
        trail: Sequence[TrailPoint],
        now: float | None = None,
    ) -> None:
        """Advance physics and resolve trail collisions. dt in seconds."""
        self.last_events.clear()
        if self.state != GameState.PLAYING:
            return

        now = time.time() if now is None else now
        self._maybe_spawn(now)
        self._integrate(dt)
        self._resolve_slices(trail)
        self._cull_missed()

    # --- spawning ------------------------------------------------------------

    def _maybe_spawn(self, now: float) -> None:
        if now < self._next_spawn_at:
            return
        self.fruits.append(self._make_projectile())
        self._next_spawn_at = now + self.spawn_interval
        self.spawn_interval = max(
            config.SPAWN_INTERVAL_MIN,
            self.spawn_interval * config.SPAWN_INTERVAL_DECAY,
        )

    def _make_projectile(self) -> Fruit:
        is_bomb = random.random() < config.BOMB_SPAWN_CHANCE
        x = random.uniform(self.width * 0.15, self.width * 0.85)
        y = float(self.height + 40)
        vx = random.uniform(-config.FRUIT_LAUNCH_VX_SPREAD, config.FRUIT_LAUNCH_VX_SPREAD)
        # Bias velocity toward center so fruit arcs into view.
        if x < self.width * 0.35:
            vx = abs(vx)
        elif x > self.width * 0.65:
            vx = -abs(vx)
        vy = random.uniform(config.FRUIT_LAUNCH_VY_MIN, config.FRUIT_LAUNCH_VY_MAX)

        if is_bomb:
            return Fruit(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                radius=float(config.BOMB_RADIUS),
                name="BOMB",
                color_bgr=(40, 40, 40),
                is_bomb=True,
            )

        name, color = random.choice(config.FRUIT_TYPES)
        radius = float(random.randint(config.FRUIT_RADIUS_MIN, config.FRUIT_RADIUS_MAX))
        return Fruit(
            x=x, y=y, vx=vx, vy=vy, radius=radius, name=name, color_bgr=color
        )

    # --- physics -------------------------------------------------------------

    def _integrate(self, dt: float) -> None:
        if dt <= 0:
            return
        g = config.GRAVITY
        for f in self.fruits:
            if not f.alive:
                continue
            f.vy += g * dt
            f.x += f.vx * dt
            f.y += f.vy * dt

    # --- collision / scoring -------------------------------------------------

    def _resolve_slices(self, trail: Sequence[TrailPoint]) -> None:
        if len(trail) < 2:
            return
        for fruit in self.fruits:
            if not fruit.alive:
                continue
            if self._trail_hits_fruit(trail, fruit):
                fruit.alive = False
                if fruit.is_bomb:
                    self._on_bomb()
                    return
                self._on_fruit_sliced(fruit)

    @staticmethod
    def _segment_circle_hit(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        cx: float,
        cy: float,
        radius: float,
    ) -> bool:
        """True if segment AB comes within `radius` of point C."""
        abx, aby = bx - ax, by - ay
        acx, acy = cx - ax, cy - ay
        ab_len2 = abx * abx + aby * aby
        if ab_len2 <= 1e-6:
            return math.hypot(acx, acy) <= radius
        t = max(0.0, min(1.0, (acx * abx + acy * aby) / ab_len2))
        px, py = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy) <= radius

    def _trail_hits_fruit(self, trail: Sequence[TrailPoint], fruit: Fruit) -> bool:
        for i in range(1, len(trail)):
            a, b = trail[i - 1], trail[i]
            dt = b.t - a.t
            if dt <= 0:
                continue
            dist = math.hypot(b.x - a.x, b.y - a.y)
            speed = dist / dt
            if speed < config.MIN_SLICE_SPEED:
                continue
            if self._segment_circle_hit(
                a.x, a.y, b.x, b.y, fruit.x, fruit.y, fruit.radius
            ):
                return True
        return False

    def _on_fruit_sliced(self, fruit: Fruit) -> None:
        now = time.time()
        if now - self._last_slice_at <= config.COMBO_WINDOW_SEC:
            self._combo += 1
        else:
            self._combo = 1
        self._last_slice_at = now
        points = config.SLICE_SCORE + max(0, self._combo - 1) * config.COMBO_BONUS
        self.score += points
        self.last_events.append(
            SliceEvent(x=fruit.x, y=fruit.y, name=fruit.name, points=points)
        )

    def _on_bomb(self) -> None:
        self.state = GameState.GAME_OVER
        self.fruits = [f for f in self.fruits if f.alive]

    def _cull_missed(self) -> None:
        """Fruits that fall off the bottom without being sliced cost a life."""
        kept: list[Fruit] = []
        for f in self.fruits:
            if not f.alive:
                continue
            if f.y - f.radius > self.height + 10:
                if not f.is_bomb:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.lives = 0
                        self.state = GameState.GAME_OVER
                # bombs that exit don't end the run
                continue
            kept.append(f)
        self.fruits = kept

    # --- test helpers --------------------------------------------------------

    def spawn_fruit_at(
        self,
        x: float,
        y: float,
        *,
        radius: float = 40.0,
        name: str = "Apple",
        color_bgr: tuple[int, int, int] = (60, 60, 220),
        is_bomb: bool = False,
    ) -> Fruit:
        """Insert a stationary fruit (for unit tests)."""
        fruit = Fruit(
            x=x,
            y=y,
            vx=0.0,
            vy=0.0,
            radius=radius,
            name=name,
            color_bgr=color_bgr,
            is_bomb=is_bomb,
        )
        self.fruits.append(fruit)
        return fruit
