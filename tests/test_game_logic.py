"""Unit tests for game logic — no camera / MediaPipe required."""

from __future__ import annotations

import time

import pytest

from fruit_ninja_cam.game import Game, GameState
from fruit_ninja_cam.hand_tracker import TrailPoint
from fruit_ninja_cam import config


def _fast_slash(x0: float, y0: float, x1: float, y1: float, t0: float | None = None) -> list[TrailPoint]:
    """Two-point trail with speed well above MIN_SLICE_SPEED."""
    t0 = time.time() if t0 is None else t0
    # distance / dt >> MIN_SLICE_SPEED (450 px/s); 200px in 0.01s = 20_000 px/s
    return [
        TrailPoint(x=x0, y=y0, t=t0),
        TrailPoint(x=x1, y=y1, t=t0 + 0.01),
    ]


def test_trail_collision_slices_fruit_and_scores() -> None:
    game = Game(width=800, height=600)
    game.start()
    fruit = game.spawn_fruit_at(400, 300, radius=40, name="Apple")

    trail = _fast_slash(360, 300, 440, 300)
    game.update(dt=0.016, trail=trail)

    assert not fruit.alive
    assert game.score == config.SLICE_SCORE
    assert game.state == GameState.PLAYING
    assert len(game.last_events) == 1
    assert game.last_events[0].name == "Apple"


def test_slow_trail_does_not_slice() -> None:
    game = Game(width=800, height=600)
    game.start()
    fruit = game.spawn_fruit_at(400, 300, radius=40)

    t0 = time.time()
    # 10px over 1.0s = 10 px/s << MIN_SLICE_SPEED
    trail = [
        TrailPoint(x=390, y=300, t=t0),
        TrailPoint(x=400, y=300, t=t0 + 1.0),
    ]
    game.update(dt=0.016, trail=trail)

    assert fruit.alive
    assert game.score == 0


def test_bomb_ends_run() -> None:
    game = Game(width=800, height=600)
    game.start()
    bomb = game.spawn_fruit_at(400, 300, radius=40, name="BOMB", is_bomb=True)

    trail = _fast_slash(360, 300, 440, 300)
    game.update(dt=0.016, trail=trail)

    assert not bomb.alive
    assert game.state == GameState.GAME_OVER


def test_miss_loses_life() -> None:
    game = Game(width=800, height=600)
    game.start()
    assert game.lives == config.STARTING_LIVES

    # Place fruit already below the bottom edge so cull removes it.
    game.spawn_fruit_at(400, game.height + 100, radius=40, name="Orange")
    game.update(dt=0.016, trail=[])

    assert game.lives == config.STARTING_LIVES - 1
    assert game.state == GameState.PLAYING
    assert game.fruits == []


def test_three_misses_game_over() -> None:
    game = Game(width=800, height=600)
    game.start()

    for _ in range(config.STARTING_LIVES):
        game.spawn_fruit_at(200, game.height + 80, radius=30)
        game.update(dt=0.016, trail=[])

    assert game.lives == 0
    assert game.state == GameState.GAME_OVER


def test_combo_bonus() -> None:
    game = Game(width=800, height=600)
    game.start()
    game.spawn_fruit_at(300, 300, radius=35)
    game.spawn_fruit_at(500, 300, radius=35)

    t0 = time.time()
    # Slash both fruits in one update with a multi-segment trail.
    trail = [
        TrailPoint(x=250, y=300, t=t0),
        TrailPoint(x=350, y=300, t=t0 + 0.01),
        TrailPoint(x=450, y=300, t=t0 + 0.02),
        TrailPoint(x=550, y=300, t=t0 + 0.03),
    ]
    game.update(dt=0.016, trail=trail)

    # First slice: SLICE_SCORE; second within combo window: SLICE_SCORE + COMBO_BONUS
    assert game.score == config.SLICE_SCORE + (config.SLICE_SCORE + config.COMBO_BONUS)
    assert sum(1 for f in game.fruits if f.alive) == 0


def test_slice_event_carries_geometry_for_effects() -> None:
    game = Game(width=800, height=600)
    game.start()
    game.spawn_fruit_at(400, 300, radius=44, name="Orange", color_bgr=(24, 138, 252))

    # Slash downward at 45 degrees through the fruit.
    game.update(dt=0.016, trail=_fast_slash(340, 240, 460, 360))

    ev = game.last_events[0]
    assert ev.radius == 44
    assert ev.color_bgr == (24, 138, 252)
    assert ev.angle_deg == pytest.approx(45.0)
    assert ev.combo == 1
    assert not ev.is_bomb


def test_bomb_emits_an_event_for_the_explosion() -> None:
    game = Game(width=800, height=600)
    game.start()
    game.spawn_fruit_at(400, 300, radius=40, name="BOMB", is_bomb=True)
    game.update(dt=0.016, trail=_fast_slash(360, 300, 440, 300))

    assert len(game.last_events) == 1
    assert game.last_events[0].is_bomb


def test_combo_expires_after_the_window() -> None:
    game = Game(width=800, height=600)
    t0 = 1000.0
    game.start(now=t0)
    game.spawn_fruit_at(400, 300, radius=40)
    game.update(dt=0.016, trail=_fast_slash(360, 300, 440, 300, t0), now=t0)
    assert game.combo == 1

    game.update(dt=0.016, trail=[], now=t0 + config.COMBO_WINDOW_SEC + 0.1)
    assert game.combo == 0


def test_start_resets_state() -> None:
    game = Game(width=800, height=600)
    game.start()
    game.score = 99
    game.lives = 1
    game.state = GameState.GAME_OVER
    game.start()
    assert game.score == 0
    assert game.lives == config.STARTING_LIVES
    assert game.state == GameState.PLAYING
    assert game.fruits == []
