"""Window, camera, and gameplay constants."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"

# --- Window / camera ---------------------------------------------------------
WINDOW_NAME = "Fruit Ninja Cam"
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MIRROR_WEBCAM = True  # selfie-style horizontal flip

# --- Hand tracking -----------------------------------------------------------
INDEX_FINGERTIP_LANDMARK = 8
TRAIL_MAX_POINTS = 16
TRAIL_MAX_AGE_SEC = 0.35
HAND_NUM_HANDS = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# --- Gameplay ----------------------------------------------------------------
STARTING_LIVES = 3
# Physics is scaled to the live frame size in game.py (not fixed 720p).
GRAVITY = 1400.0  # px / s^2 at CAMERA_HEIGHT reference
SPAWN_INTERVAL_START = 1.1  # seconds between spawns
SPAWN_INTERVAL_MIN = 0.45
SPAWN_INTERVAL_DECAY = 0.985  # multiply interval after each spawn
BOMB_SPAWN_CHANCE = 0.18
FRUIT_RADIUS_MIN = 36
FRUIT_RADIUS_MAX = 52
BOMB_RADIUS = 40
# Peak height as a fraction of frame height (0 = top). Fruit aims between these.
FRUIT_APEX_Y_FRAC_MIN = 0.04
FRUIT_APEX_Y_FRAC_MAX = 0.18
FRUIT_LAUNCH_VX_SPREAD_FRAC = 0.22  # ± fraction of frame width
# Kept for tests / fallback docs; live spawns use apex fractions above.
FRUIT_LAUNCH_VY_MIN = -820.0
FRUIT_LAUNCH_VY_MAX = -560.0
FRUIT_LAUNCH_VX_SPREAD = 280.0
# Minimum tip speed (px/s) along a trail segment to count as a slice.
MIN_SLICE_SPEED = 450.0
SLICE_SCORE = 10
COMBO_WINDOW_SEC = 0.6
COMBO_BONUS = 5

FRUIT_TYPES: list[tuple[str, tuple[int, int, int]]] = [
    ("Apple", (48, 46, 214)),      # BGR
    ("Orange", (24, 138, 252)),
    ("Banana", (72, 214, 244)),
    ("Watermelon", (62, 148, 58)),
    ("Grape", (156, 54, 138)),
    ("Lemon", (52, 216, 246)),
]

# --- Presentation ------------------------------------------------------------
# The webcam feed is the backdrop, so it is pushed back (dimmed, desaturated,
# vignetted) to keep the fruit and blade legible over any room lighting.
BACKDROP_DIM = 0.34          # 0 = untouched feed, 1 = black
BACKDROP_DESATURATE = 0.30   # 0 = full colour, 1 = greyscale
VIGNETTE_STRENGTH = 0.55
SHOW_FRUIT_LABELS = False    # sprites are readable without name tags
FRUIT_SHADOW = True
BLADE_GLOW = True
BLADE_WIDTH = 15             # px at the tip, tapering to the trail's tail
SCREEN_SHAKE = True
HUD_ACCENT = (86, 196, 250)  # BGR amber
HUD_DANGER = (72, 68, 255)
