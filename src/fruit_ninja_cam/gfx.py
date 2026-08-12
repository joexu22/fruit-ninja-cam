"""Low-level compositing helpers: alpha blits, glow, panels, text."""

from __future__ import annotations

import cv2
import numpy as np

BGR = tuple[int, int, int]

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_HEAVY = cv2.FONT_HERSHEY_TRIPLEX


def _overlap(
    frame: np.ndarray, size_hw: tuple[int, int], x: int, y: int
) -> tuple[slice, slice, slice, slice] | None:
    """Clip a sprite of size (h, w) placed at top-left (x, y) against the frame."""
    fh, fw = frame.shape[:2]
    sh, sw = size_hw
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(fw, x + sw), min(fh, y + sh)
    if dx0 >= dx1 or dy0 >= dy1:
        return None
    return (
        slice(dy0, dy1),
        slice(dx0, dx1),
        slice(dy0 - y, dy1 - y),
        slice(dx0 - x, dx1 - x),
    )


def blit(frame: np.ndarray, bgra: np.ndarray, cx: float, cy: float, opacity: float = 1.0) -> None:
    """Alpha-composite a BGRA sprite centred on (cx, cy)."""
    if opacity <= 0.003:
        return
    x = int(round(cx - bgra.shape[1] / 2))
    y = int(round(cy - bgra.shape[0] / 2))
    reg = _overlap(frame, bgra.shape[:2], x, y)
    if reg is None:
        return
    dy, dx, sy, sx = reg
    src = bgra[sy, sx]
    a = src[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    roi = frame[dy, dx]
    roi[:] = (src[:, :, :3].astype(np.float32) * a + roi.astype(np.float32) * (1.0 - a)).astype(
        np.uint8
    )


def blit_add(frame: np.ndarray, bgr: np.ndarray, cx: float, cy: float, gain: float = 1.0) -> None:
    """Additively blend a BGR layer centred on (cx, cy) — for glows and sparks."""
    if gain <= 0.003:
        return
    x = int(round(cx - bgr.shape[1] / 2))
    y = int(round(cy - bgr.shape[0] / 2))
    reg = _overlap(frame, bgr.shape[:2], x, y)
    if reg is None:
        return
    dy, dx, sy, sx = reg
    roi = frame[dy, dx]
    roi[:] = cv2.add(roi, (bgr[sy, sx].astype(np.float32) * gain).astype(np.uint8))


def rotated(bgra: np.ndarray, degrees: float, scale: float = 1.0) -> np.ndarray:
    """Rotate a BGRA sprite about its centre, keeping the canvas size."""
    if abs(degrees) < 0.5 and abs(scale - 1.0) < 0.01:
        return bgra
    h, w = bgra.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, scale)
    return cv2.warpAffine(
        bgra, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def radial_glow(radius: int, color: BGR, falloff: float = 2.2) -> np.ndarray:
    """A soft additive glow disc of the given radius."""
    size = max(3, radius * 2 + 1)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / max(1.0, c)
    falloff_mask = np.clip(1.0 - d, 0.0, 1.0) ** falloff
    return (falloff_mask[..., None] * np.array(color, np.float32)).astype(np.uint8)


def vignette_mask(h: int, w: int, strength: float = 0.45) -> np.ndarray:
    """uint8 multiplier map (255 = untouched) darkening the frame edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2.0) / (w / 2.0)
    ny = (yy - h / 2.0) / (h / 2.0)
    d = np.sqrt(nx * nx + ny * ny) / 1.4142
    k = 1.0 - strength * np.clip(d, 0.0, 1.0) ** 1.9
    return np.clip(k * 255.0, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)


def rounded_rect_mask(w: int, h: int, radius: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    r = max(0, min(radius, min(w, h) // 2))
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    radius: int = 16,
    color: BGR = (26, 22, 30),
    alpha: float = 0.55,
    border: BGR | None = None,
    border_alpha: float = 0.5,
) -> None:
    """Frosted rounded panel: darkens and slightly blurs what is behind it."""
    reg = _overlap(frame, (h, w), x, y)
    if reg is None:
        return
    dy, dx, sy, sx = reg
    mask = rounded_rect_mask(w, h, radius)[sy, sx].astype(np.float32) / 255.0
    roi = frame[dy, dx].astype(np.float32)
    blurred = cv2.GaussianBlur(frame[dy, dx], (0, 0), 6.0).astype(np.float32)
    tint = np.array(color, np.float32)
    filled = blurred * (1.0 - alpha) + tint * alpha
    m = mask[..., None]
    frame[dy, dx] = (roi * (1.0 - m) + filled * m).astype(np.uint8)

    if border is not None:
        edge = cv2.morphologyEx(
            rounded_rect_mask(w, h, radius), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )[sy, sx].astype(np.float32) / 255.0
        e = (edge * border_alpha)[..., None]
        base = frame[dy, dx].astype(np.float32)
        frame[dy, dx] = (base * (1 - e) + np.array(border, np.float32) * e).astype(np.uint8)


def blend_mask(roi: np.ndarray, mask: np.ndarray, color: BGR, opacity: float = 1.0) -> None:
    a = (mask.astype(np.float32) * (opacity / 255.0))[..., None]
    roi[:] = (np.array(color, np.float32) * a + roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def text(
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    *,
    scale: float = 0.8,
    color: BGR = (255, 255, 255),
    thickness: int = 2,
    font: int = FONT,
    shadow: bool = True,
    center: bool = False,
    outline: BGR | None = None,
) -> tuple[int, int]:
    """Draw text with a soft drop shadow and optional outline.

    Composited through a stencil rather than by stacking thicker `putText`
    passes: Hershey glyph advance grows with stroke thickness, so stacked
    passes drift apart and ghost at the end of long strings.
    """
    (tw, th), base = cv2.getTextSize(label, font, scale, thickness)
    if center:
        x -= tw // 2
    grow = 3 if outline is not None else 0
    pad = 8 + thickness + grow
    stencil = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
    cv2.putText(stencil, label, (pad, pad + th), font, scale, 255, thickness, cv2.LINE_AA)

    reg = _overlap(frame, stencil.shape[:2], x - pad, y - th - pad)
    if reg is None:
        return tw, th
    dy, dx, sy, sx = reg
    roi = frame[dy, dx]

    if shadow:
        drop = np.zeros_like(stencil)
        drop[3:, 2:] = stencil[:-3, :-2]
        drop = cv2.GaussianBlur(drop, (0, 0), 2.0 + thickness * 0.5)
        blend_mask(roi, drop[sy, sx], (0, 0, 0), 0.85)
    if outline is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        blend_mask(roi, cv2.dilate(stencil, k)[sy, sx], outline)
    blend_mask(roi, stencil[sy, sx], color)
    return tw, th


def text_width(label: str, scale: float, thickness: int = 2, font: int = FONT) -> int:
    return int(cv2.getTextSize(label, font, scale, thickness)[0][0])


def heart(frame: np.ndarray, cx: int, cy: int, size: int, color: BGR, filled: bool) -> None:
    """A small vector heart — OpenCV's Hershey fonts have no glyph for it."""
    r = max(2, size // 2)
    pts = np.array(
        [
            [cx, cy + size],
            [cx - int(size * 1.02), cy - int(size * 0.10)],
            [cx - int(size * 0.52), cy - int(size * 0.72)],
            [cx, cy - int(size * 0.24)],
            [cx + int(size * 0.52), cy - int(size * 0.72)],
            [cx + int(size * 1.02), cy - int(size * 0.10)],
        ],
        np.int32,
    )
    if filled:
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
        cv2.circle(frame, (cx - r // 2, cy - int(size * 0.34)), max(1, r // 3), (255, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)


def tint(frame: np.ndarray, color: BGR, amount: float) -> None:
    """Blend the whole frame toward a colour, in place and without allocating."""
    amount = max(0.0, min(1.0, amount))
    if amount <= 0.003:
        return
    cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - amount)
    cv2.add(frame, (color[0] * amount, color[1] * amount, color[2] * amount, 0.0), dst=frame)


def bloom(frame: np.ndarray, layer: np.ndarray, sigma: float = 9.0, gain: float = 1.0) -> None:
    """Blur a full-frame additive layer at reduced resolution and add it back."""
    h, w = frame.shape[:2]
    small = cv2.resize(layer, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 3.0)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if gain != 1.0:
        up = (up.astype(np.float32) * gain).astype(np.uint8)
    cv2.add(frame, up, frame)
