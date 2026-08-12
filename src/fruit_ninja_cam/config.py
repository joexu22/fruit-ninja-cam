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
GRAVITY = 1400.0  # px / s^2
SPAWN_INTERVAL_START = 1.1  # seconds between spawns
SPAWN_INTERVAL_MIN = 0.45
SPAWN_INTERVAL_DECAY = 0.985  # multiply interval after each spawn
BOMB_SPAWN_CHANCE = 0.18
FRUIT_RADIUS_MIN = 36
FRUIT_RADIUS_MAX = 52
BOMB_RADIUS = 40
FRUIT_LAUNCH_VY_MIN = -820.0
FRUIT_LAUNCH_VY_MAX = -560.0
FRUIT_LAUNCH_VX_SPREAD = 280.0
# Minimum tip speed (px/s) along a trail segment to count as a slice.
MIN_SLICE_SPEED = 450.0
SLICE_SCORE = 10
COMBO_WINDOW_SEC = 0.6
COMBO_BONUS = 5

FRUIT_TYPES: list[tuple[str, tuple[int, int, int]]] = [
    ("Apple", (60, 60, 220)),      # BGR
    ("Orange", (0, 140, 255)),
    ("Banana", (0, 220, 240)),
    ("Watermelon", (80, 180, 60)),
    ("Grape", (180, 60, 160)),
]
