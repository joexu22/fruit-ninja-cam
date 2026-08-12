"""Smoke tests for the presentation layer — no camera or GPU required."""

from __future__ import annotations

import numpy as np

from fruit_ninja_cam import config, theme
from fruit_ninja_cam.effects import Effects
from fruit_ninja_cam.game import Game, GameState
from fruit_ninja_cam.hand_tracker import TrailPoint
from fruit_ninja_cam.render import Renderer

W, H = 320, 240


def _frame() -> np.ndarray:
    return np.full((H, W, 3), 60, np.uint8)


def _slash(x0: float, y0: float, x1: float, y1: float, t0: float = 1000.0) -> list[TrailPoint]:
    return [TrailPoint(x=x0, y=y0, t=t0), TrailPoint(x=x1, y=y1, t=t0 + 0.01)]


def test_render_returns_new_frame_and_leaves_input_untouched() -> None:
    frame = _frame()
    original = frame.copy()
    game = Game(width=W, height=H)
    out = Renderer().render(frame, game, [], None, dt=1 / 30)

    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert np.array_equal(frame, original)


def test_every_game_state_renders() -> None:
    renderer = Renderer()
    game = Game(width=W, height=H)
    trail = _slash(10, 10, 120, 90)
    for state in GameState:
        game.state = state
        out = renderer.render(_frame(), game, trail, (120.0, 90.0), dt=1 / 30)
        assert out.shape == (H, W, 3)


def test_slice_event_spawns_halves_and_juice() -> None:
    renderer = Renderer()
    game = Game(width=W, height=H)
    game.start(now=1000.0)
    game.spawn_fruit_at(160, 120, radius=30, name="Apple")
    game.update(dt=1 / 30, trail=_slash(120, 120, 200, 120), now=1000.1)

    renderer.render(_frame(), game, [], None, dt=1 / 30)
    assert len(renderer.effects.halves) == 2
    assert renderer.effects.particles


def test_effects_are_consumed_once_while_the_game_is_frozen() -> None:
    """GAME_OVER stops calling update(), so events must not re-fire each frame."""
    renderer = Renderer()
    game = Game(width=W, height=H)
    game.start(now=1000.0)
    game.spawn_fruit_at(160, 120, radius=30, name="BOMB", is_bomb=True)
    game.update(dt=1 / 30, trail=_slash(120, 120, 200, 120), now=1000.1)
    assert game.state == GameState.GAME_OVER

    renderer.render(_frame(), game, [], None, dt=1 / 30)
    after_first = len(renderer.effects.particles)
    for _ in range(3):
        renderer.render(_frame(), game, [], None, dt=0.0)
    assert len(renderer.effects.particles) <= after_first


def test_effects_retire_over_time() -> None:
    fx = Effects(seed=1)
    fx.slice_fruit(
        name="Orange", radius=30, skin=(24, 138, 252), x=100, y=100, vx=0, vy=0, angle_deg=15
    )
    assert fx.halves and fx.particles
    for _ in range(120):
        fx.update(0.05, W, H)
    assert not fx.halves
    assert not fx.particles


def test_sprites_are_baked_with_transparency() -> None:
    for name, color in config.FRUIT_TYPES:
        sprite = theme.fruit_sprite(name, 40, color)
        assert sprite.bgra.shape[2] == 4
        alpha = sprite.bgra[:, :, 3]
        assert alpha.max() == 255, name  # the fruit is opaque somewhere
        assert alpha[0, 0] == 0, name  # and the corners stay transparent
