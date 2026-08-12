# Fruit Ninja Cam

**Webcam Fruit Ninja** powered by [Google MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker). Your **index fingertip** (landmark 8) is the blade — swipe to slice falling fruit, dodge bombs, keep 3 lives.

Mac-first scaffold (OpenCV + MediaPipe Python Tasks). The same Hand Landmarker `.task` model runs on **Android / iOS / Web** via MediaPipe Tasks — this repo is a clean Python reference you can port.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![MediaPipe Tasks](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-green)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **Index-fingertip blade trail** — MediaPipe Tasks `HandLandmarker` in `VIDEO` mode
- **Mirrored webcam** (selfie view) so motion feels natural
- **Fruit + bombs**, gravity arcs, combo scoring, 3 lives
- **SPACE** start / restart · **Q** (or Esc) quit
- Model downloaded by script (not committed) — idempotent `scripts/download_models.py`
- Unit-tested game logic (collision, scoring, bomb game-over, missed fruit) — no camera required

## Quickstart (macOS)

```bash
# 1. Clone
git clone https://github.com/joexu22/fruit-ninja-cam.git
cd fruit-ninja-cam

# 2. Create a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install package + download hand_landmarker.task
make setup

# 4. Play
make run
# or: python -m fruit_ninja_cam
```

### macOS camera permission

The first launch will prompt for camera access. If the window stays black:

1. **System Settings → Privacy & Security → Camera**
2. Enable **Terminal**, **iTerm2**, or your IDE (VS Code / Cursor / PyCharm)
3. Quit and relaunch the app (permission changes apply to new processes)

### Tests

```bash
make test
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start game / restart after game over |
| `Q` / `Esc` | Quit |

Swipe quickly through fruit with your **index finger**. Hitting a **bomb** ends the run. Letting fruit fall off the bottom costs a life.

## How it works

```
Webcam frame ──flip──► HandTracker (MediaPipe landmark 8)
                              │
                              ▼  timed tip trail
                         Game.update (gravity, spawn, segment↔circle slice)
                              │
                              ▼
                         render overlays ──► imshow
```

1. **`hand_tracker.py`** — MediaPipe Tasks Vision `HandLandmarker` (`RunningMode.VIDEO`). Converts normalized landmark 8 → pixels; keeps a short timestamped trail. Falls back to legacy `mp.solutions.hands` if Tasks imports are unavailable.
2. **`game.py`** — Pure logic: `Fruit` / bomb dataclasses, spawn cadence, gravity, trail-segment vs circle collision with a **minimum slice speed**, score / lives / `MENU | PLAYING | GAME_OVER`.
3. **`render.py`** — Colored fruit circles + labels, bomb fuse, blade trail, HUD.
4. **`main.py`** — Camera loop wiring tracker + game + render.

### Model

Official Google float16 bundle (not in git):

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Saved to `models/hand_landmarker.task` by `scripts/download_models.py`.

## Project layout

```
fruit-ninja-cam/
  README.md
  LICENSE
  pyproject.toml
  Makefile
  .gitignore
  scripts/download_models.py
  src/fruit_ninja_cam/
    __init__.py
    __main__.py
    main.py
    hand_tracker.py
    game.py
    render.py
    config.py
  tests/test_game_logic.py
```

## Android-ready note

This demo uses the same **MediaPipe Tasks Hand Landmarker** model and landmark indices as Google’s Android / iOS samples. A mobile port can reuse:

- Landmark **8** as the blade tip
- Segment–circle slice tests against projected fruit
- The downloaded `hand_landmarker.task` asset (or the Gradle download task from [mediapipe-samples](https://github.com/google-ai-edge/mediapipe-samples))

Python here is the fastest way to iterate on feel; MediaPipe keeps inference portable.

## Interview / demo talking points

- End-to-end **ML → interaction → game loop** with a production Google model
- **Tasks API** (not only legacy solutions), VIDEO timestamps, trail-based slash detection
- Separated **pure game logic** (unit-tested) from I/O and rendering
- Mac-first DX: Makefile, editable install, camera permission docs

## Requirements

- Python 3.10+
- Webcam
- Dependencies: `mediapipe`, `opencv-python`, `numpy` (see `pyproject.toml`)

## License

MIT © 2026 Joe Xu — see [LICENSE](LICENSE).
