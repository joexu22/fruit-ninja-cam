#!/usr/bin/env python3
"""Download the official MediaPipe Hand Landmarker .task model into models/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Official Google storage URL (MediaPipe Tasks vision Hand Landmarker float16).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"


def download_model(force: bool = False) -> Path:
    """Download hand_landmarker.task if missing (idempotent)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not force:
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Model already present: {MODEL_PATH} ({size_mb:.1f} MB). Skipping download.")
        return MODEL_PATH

    print(f"Downloading Hand Landmarker model…\n  {MODEL_URL}")
    tmp_path = MODEL_PATH.with_suffix(".task.partial")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        tmp_path.replace(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"ERROR: failed to download model: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved to {MODEL_PATH} ({size_mb:.1f} MB).")
    return MODEL_PATH


def main() -> None:
    force = "--force" in sys.argv
    download_model(force=force)


if __name__ == "__main__":
    main()
