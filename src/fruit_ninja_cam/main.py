"""Camera loop — MediaPipe hand tracking + Fruit Ninja game + render."""

from __future__ import annotations

import sys
import time

import cv2

from fruit_ninja_cam import config
from fruit_ninja_cam.game import Game, GameState
from fruit_ninja_cam.hand_tracker import HandTracker
from fruit_ninja_cam.render import draw_frame


def main() -> int:
    if not config.MODEL_PATH.exists():
        # Tasks API needs the model; download script is the supported path.
        print(
            f"Model missing: {config.MODEL_PATH}\n"
            "Run `make setup` or `python scripts/download_models.py` first.",
            file=sys.stderr,
        )
        # Still attempt — HandTracker may fall back to legacy solutions.
        # If Tasks is available without model, it will raise a clear error.

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(
            f"ERROR: cannot open camera index {config.CAMERA_INDEX}.\n"
            "On macOS: System Settings → Privacy & Security → Camera → "
            "allow Terminal / iTerm / your IDE.",
            file=sys.stderr,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    game = Game(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.CAMERA_WIDTH),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.CAMERA_HEIGHT),
    )

    print("Fruit Ninja Cam — SPACE start/restart, Q quit. Mirror on.")
    print("Slice fruit with your index fingertip. Avoid bombs!")

    try:
        with HandTracker() as tracker:
            return _run_loop(cap, game, tracker)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _run_loop(cap: cv2.VideoCapture, game: Game, tracker: HandTracker) -> int:
    prev_t = time.time()
    # MediaPipe VIDEO mode requires monotonically increasing timestamps (ms).
    t0_ms = int(prev_t * 1000)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: failed to read frame from camera.", file=sys.stderr)
            return 1

        if config.MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)

        # Keep game canvas size in sync with actual frames.
        game.height, game.width = frame.shape[:2]

        now = time.time()
        dt = min(0.05, max(0.0, now - prev_t))
        prev_t = now
        timestamp_ms = int(now * 1000) - t0_ms
        if timestamp_ms < 0:
            timestamp_ms = 0

        tip = tracker.process(frame, timestamp_ms)
        trail = tracker.trail

        if game.state == GameState.PLAYING:
            game.update(dt, trail, now=now)

        display = draw_frame(frame, game, trail, tip)
        cv2.imshow(config.WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # q / Esc
            break
        if key == ord(" "):
            if game.state in (GameState.MENU, GameState.GAME_OVER):
                tracker.clear_trail()
                game.start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
