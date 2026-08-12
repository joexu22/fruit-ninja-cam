"""MediaPipe Hand Landmarker wrapper — index fingertip + timed trail."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from fruit_ninja_cam import config

try:
    # Preferred: MediaPipe Tasks Vision API
    from mediapipe.tasks.python.core import base_options as mp_base_options
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    import mediapipe as mp

    _USE_TASKS_API = True
except ImportError:  # pragma: no cover - environment-dependent
    # Fallback: legacy mp.solutions.hands (older mediapipe wheels)
    import mediapipe as mp

    _USE_TASKS_API = False


@dataclass(frozen=True)
class TrailPoint:
    """Pixel position of the index fingertip with a wall-clock timestamp."""

    x: float
    y: float
    t: float  # time.time()


class HandTracker:
    """Track the index fingertip (landmark 8) and keep a short timed trail."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = Path(model_path or config.MODEL_PATH)
        self._trail: Deque[TrailPoint] = deque(maxlen=config.TRAIL_MAX_POINTS)
        self._landmarker: object | None = None
        self._legacy_hands: object | None = None
        self._init_detector()

    def _init_detector(self) -> None:
        if _USE_TASKS_API:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Hand Landmarker model not found at {self._model_path}. "
                    "Run: python scripts/download_models.py  (or `make setup`)"
                )
            options = HandLandmarkerOptions(
                base_options=mp_base_options.BaseOptions(
                    model_asset_path=str(self._model_path)
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=config.HAND_NUM_HANDS,
                min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.HAND_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        else:
            # Legacy solutions API — no .task file required.
            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=config.HAND_NUM_HANDS,
                min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
            )

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> Optional[tuple[float, float]]:
        """
        Detect the index fingertip in a BGR frame.

        Returns pixel (x, y) in the given frame's coordinate space, or None.
        Updates the internal trail. Caller should pass the (optionally mirrored)
        frame that will be displayed.
        """
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()  # BGR → RGB, contiguous
        tip: Optional[tuple[float, float]] = None

        if _USE_TASKS_API and self._landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.hand_landmarks:
                lm = result.hand_landmarks[0][config.INDEX_FINGERTIP_LANDMARK]
                tip = (lm.x * w, lm.y * h)
        elif self._legacy_hands is not None:
            result = self._legacy_hands.process(rgb)
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark[
                    config.INDEX_FINGERTIP_LANDMARK
                ]
                tip = (lm.x * w, lm.y * h)

        now = time.time()
        self._prune_trail(now)
        if tip is not None:
            self._trail.append(TrailPoint(x=tip[0], y=tip[1], t=now))
        return tip

    def _prune_trail(self, now: float) -> None:
        cutoff = now - config.TRAIL_MAX_AGE_SEC
        while self._trail and self._trail[0].t < cutoff:
            self._trail.popleft()

    @property
    def trail(self) -> list[TrailPoint]:
        """Copy of recent tip positions (oldest → newest)."""
        now = time.time()
        self._prune_trail(now)
        return list(self._trail)

    def clear_trail(self) -> None:
        self._trail.clear()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._legacy_hands is not None:
            self._legacy_hands.close()
            self._legacy_hands = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
