"""Visual identity: palette, per-fruit art styles, and cached sprite baking.

Sprites are baked once per (kind, radius) at import-time cost only, then
alpha-blitted every frame. Shading is derived generically from a silhouette
mask: a distance transform fakes a rounded surface, whose gradient gives
normals for Lambert + specular + rim lighting. That keeps one shading path for
spheres, crescents (banana) and clusters (grapes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

BGR = tuple[int, int, int]

# --- Palette -----------------------------------------------------------------

INK = (18, 16, 26)
PAPER = (242, 244, 248)
GOLD = (86, 196, 250)
BLADE_CORE = (255, 255, 255)
BLADE_GLOW = (255, 190, 90)
DANGER = (72, 68, 255)
MINT = (150, 240, 190)

# Light direction (x right, y down, z toward viewer), upper-left key light.
LIGHT_DIR = (-0.42, -0.58, 0.70)
SUPERSAMPLE = 3


@dataclass(frozen=True)
class FruitStyle:
    """Art recipe for one fruit kind."""

    skin: BGR
    skin_shadow: BGR
    flesh: BGR
    juice: BGR
    shape: str = "sphere"  # sphere | crescent | cluster | ellipse
    core: BGR | None = None  # inner flesh ring (watermelon heart, citrus pith)
    rind: BGR | None = None  # thin ring just inside the skin on a cut face
    seeds: bool = False
    segments: int = 0  # citrus wedges drawn on the cut face
    stem: bool = False
    leaf: bool = False
    speckle: float = 0.0  # 0..1 skin texture strength
    stripes: BGR | None = None
    gloss: float = 0.55


STYLES: dict[str, FruitStyle] = {
    "Apple": FruitStyle(
        skin=(48, 46, 214),
        skin_shadow=(30, 24, 128),
        flesh=(206, 238, 250),
        juice=(90, 96, 236),
        core=(178, 214, 236),
        rind=(60, 60, 216),
        seeds=True,
        stem=True,
        leaf=True,
        gloss=0.75,
    ),
    "Orange": FruitStyle(
        skin=(24, 138, 252),
        skin_shadow=(12, 84, 176),
        flesh=(96, 186, 255),
        juice=(40, 158, 255),
        core=(150, 216, 255),
        rind=(190, 232, 255),
        segments=8,
        speckle=0.5,
        stem=True,
        gloss=0.45,
    ),
    "Banana": FruitStyle(
        skin=(72, 214, 244),
        skin_shadow=(30, 140, 186),
        flesh=(186, 238, 250),
        juice=(120, 226, 246),
        shape="crescent",
        rind=(70, 200, 232),
        gloss=0.5,
    ),
    "Watermelon": FruitStyle(
        skin=(62, 148, 58),
        skin_shadow=(28, 82, 32),
        flesh=(196, 236, 244),
        juice=(72, 70, 232),
        core=(70, 66, 224),
        rind=(96, 190, 118),
        stripes=(34, 96, 40),
        seeds=True,
        gloss=0.6,
    ),
    "Grape": FruitStyle(
        skin=(156, 54, 138),
        skin_shadow=(96, 26, 84),
        flesh=(178, 226, 214),
        juice=(168, 62, 150),
        shape="cluster",
        rind=(150, 70, 140),
        stem=True,
        gloss=0.8,
    ),
    "Lemon": FruitStyle(
        skin=(52, 216, 246),
        skin_shadow=(24, 140, 178),
        flesh=(150, 240, 252),
        juice=(90, 226, 248),
        shape="ellipse",
        core=(196, 246, 254),
        rind=(206, 248, 254),
        segments=7,
        speckle=0.4,
        gloss=0.5,
    ),
}

_DEFAULT_STYLE = STYLES["Apple"]


def style_for(name: str, color_bgr: BGR | None = None) -> FruitStyle:
    """Look up a style, synthesising a plausible one for unknown fruit names."""
    style = STYLES.get(name)
    if style is not None:
        return style
    if color_bgr is None:
        return _DEFAULT_STYLE
    return FruitStyle(
        skin=color_bgr,
        skin_shadow=_scale(color_bgr, 0.5),
        flesh=_mix(color_bgr, (255, 255, 255), 0.65),
        juice=color_bgr,
        rind=_scale(color_bgr, 0.85),
    )


# --- Colour helpers ----------------------------------------------------------


def _scale(c: BGR, k: float) -> BGR:
    return tuple(int(max(0, min(255, v * k))) for v in c)  # type: ignore[return-value]


def _mix(a: BGR, b: BGR, t: float) -> BGR:
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))  # type: ignore[return-value]


# --- Shading -----------------------------------------------------------------


def _normals_from_mask(mask: np.ndarray, bulge: float) -> np.ndarray:
    """Fake surface normals for a silhouette: distance transform -> dome -> gradient."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(dist / max(1.0, bulge), 0.0, 1.0)
    height = np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0)).astype(np.float32)
    height = cv2.GaussianBlur(height, (0, 0), max(0.8, bulge * 0.08))

    slope = max(1.0, bulge) * 0.55
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) * slope
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) * slope
    nz = np.ones_like(height)
    normals = np.dstack([-gx, -gy, nz])
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    return (normals / np.maximum(norm, 1e-6)).astype(np.float32)


def _shade(
    mask: np.ndarray,
    albedo: np.ndarray,
    bulge: float,
    gloss: float,
    ambient: float = 0.42,
    rim: float = 0.42,
) -> np.ndarray:
    """Light an albedo image through a silhouette mask. Returns float32 BGR."""
    normals = _normals_from_mask(mask, bulge)
    light = np.array(LIGHT_DIR, dtype=np.float32)
    light /= np.linalg.norm(light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = light + view
    half /= np.linalg.norm(half)

    ndl = np.clip(normals @ light, 0.0, 1.0)[..., None]
    ndh = np.clip(normals @ half, 0.0, 1.0)[..., None]
    ndv = np.clip(normals @ view, 0.0, 1.0)[..., None]

    diffuse = ambient + (1.0 - ambient) * ndl
    lit = albedo * diffuse
    lit += 255.0 * gloss * np.power(ndh, 42.0)
    lit += 255.0 * rim * np.power(1.0 - ndv, 3.5) * 0.55
    return np.clip(lit, 0, 255).astype(np.float32)


# --- Silhouettes -------------------------------------------------------------


def _sphere_mask(size: int, r: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(r), 255, -1, cv2.LINE_AA)
    return m


def _ellipse_mask(size: int, r: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.ellipse(
        m, (size // 2, size // 2), (int(r), int(r * 0.78)), -18, 0, 360, 255, -1, cv2.LINE_AA
    )
    # Citrus nubs at both tips.
    for sign in (-1, 1):
        px = int(size // 2 + sign * r * 0.95 * math.cos(math.radians(-18)))
        py = int(size // 2 + sign * r * 0.95 * math.sin(math.radians(-18)))
        cv2.circle(m, (px, py), int(r * 0.16), 255, -1, cv2.LINE_AA)
    return m


def _crescent_mask(size: int, r: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    c = size // 2
    cv2.ellipse(
        m,
        (c, int(c - r * 0.30)),
        (int(r * 0.98), int(r * 0.92)),
        0,
        28,
        152,
        255,
        int(r * 0.52),
        cv2.LINE_AA,
    )
    return m


_CLUSTER_OFFSETS = (
    (0.00, -0.52, 0.42),
    (-0.46, -0.16, 0.44),
    (0.46, -0.16, 0.44),
    (-0.24, 0.34, 0.46),
    (0.24, 0.34, 0.46),
    (0.00, -0.02, 0.46),
    (0.00, 0.72, 0.34),
)


def _cluster_mask(size: int, r: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    c = size // 2
    for dx, dy, br in _CLUSTER_OFFSETS:
        cv2.circle(
            m, (int(c + dx * r), int(c + dy * r)), int(br * r), 255, -1, cv2.LINE_AA
        )
    return m


def _mask_for(shape: str, size: int, r: float) -> np.ndarray:
    if shape == "crescent":
        return _crescent_mask(size, r)
    if shape == "cluster":
        return _cluster_mask(size, r)
    if shape == "ellipse":
        return _ellipse_mask(size, r)
    return _sphere_mask(size, r)


# --- Albedo ------------------------------------------------------------------


def _albedo_for(style: FruitStyle, size: int, r: float) -> np.ndarray:
    c = size // 2
    albedo = np.zeros((size, size, 3), np.float32)
    albedo[:] = np.array(style.skin, np.float32)

    if style.stripes is not None:
        stripe = np.array(style.stripes, np.float32)
        layer = np.zeros((size, size), np.uint8)
        for i in range(-2, 3):
            off = i * r * 0.46
            cv2.ellipse(
                layer,
                (int(c + off), c),
                (max(2, int(r * 0.13)), int(r * 1.05)),
                float(off / max(1.0, r) * 9.0),
                0,
                360,
                255,
                -1,
                cv2.LINE_AA,
            )
        layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, r * 0.02))
        sel = (layer.astype(np.float32) / 255.0)[..., None]
        albedo = albedo * (1 - sel) + stripe * sel

    if style.speckle > 0:
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), max(1.0, r * 0.035))
        noise /= max(1e-6, float(np.abs(noise).max()))
        albedo *= 1.0 + noise[..., None] * (0.16 * style.speckle)

    if style.shape == "cluster":
        # Per-berry tint plus a dark crease so the cluster reads as many grapes,
        # not one purple blob (the distance-transform shading alone merges them).
        tint = np.zeros((size, size, 3), np.float32)
        for i, (dx, dy, br) in enumerate(_CLUSTER_OFFSETS):
            k = 1.0 + (0.14 if i % 2 else -0.12)
            cv2.circle(
                tint, (int(c + dx * r), int(c + dy * r)), int(br * r), (k, k, k), -1, cv2.LINE_AA
            )
        tint[tint == 0] = 1.0
        for dx, dy, br in _CLUSTER_OFFSETS:
            cv2.circle(
                tint,
                (int(c + dx * r), int(c + dy * r)),
                int(br * r),
                (0.42, 0.42, 0.42),
                max(1, int(r * 0.05)),
                cv2.LINE_AA,
            )
        albedo *= tint

    if style.shape == "crescent":
        # Darker, bruised tips on the banana.
        tips = np.zeros((size, size, 3), np.float32)
        tips[:] = 1.0
        for ang in (28, 152):
            px = int(c + r * 0.98 * math.cos(math.radians(-ang)))
            py = int(c - r * 0.30 - r * 0.92 * math.sin(math.radians(ang)))
            cv2.circle(tips, (px, py), int(r * 0.30), (0.42, 0.44, 0.46), -1, cv2.LINE_AA)
        tips = cv2.GaussianBlur(tips, (0, 0), max(1.0, r * 0.10))
        albedo *= tips

    return albedo


def _draw_stem(canvas: np.ndarray, size: int, r: float, leaf: bool) -> None:
    c = size // 2
    top = int(c - r * 0.92)
    cv2.line(
        canvas,
        (c + int(r * 0.02), top + int(r * 0.16)),
        (c - int(r * 0.10), top - int(r * 0.34)),
        (26, 52, 82),
        max(2, int(r * 0.11)),
        cv2.LINE_AA,
    )
    if leaf:
        cv2.ellipse(
            canvas,
            (c + int(r * 0.30), top - int(r * 0.24)),
            (int(r * 0.34), int(r * 0.16)),
            -28,
            0,
            360,
            (46, 156, 74),
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            canvas,
            (c + int(r * 0.30), top - int(r * 0.24)),
            (int(r * 0.34), int(r * 0.16)),
            -28,
            0,
            360,
            (30, 116, 52),
            max(1, int(r * 0.05)),
            cv2.LINE_AA,
        )


# --- Sprite baking -----------------------------------------------------------


@dataclass(frozen=True)
class Sprite:
    """Baked BGRA art. Treat as immutable: instances are cached and shared."""

    bgra: np.ndarray

    @property
    def size(self) -> int:
        return int(self.bgra.shape[0])


def _finish(rgb: np.ndarray, mask: np.ndarray, out_size: int) -> Sprite:
    bgra = np.dstack([rgb, mask.astype(np.float32)]).astype(np.float32)
    bgra = cv2.resize(bgra, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return Sprite(bgra=np.clip(bgra, 0, 255).astype(np.uint8))


@lru_cache(maxsize=512)
def fruit_sprite(name: str, radius: int, skin: BGR | None = None) -> Sprite:
    """Bake a lit fruit sprite. Cached; never mutate the returned array."""
    style = style_for(name, skin)
    s = SUPERSAMPLE
    pad = 0.42
    out_size = int(radius * 2 * (1 + pad))
    size = out_size * s
    r = radius * s

    mask = _mask_for(style.shape, size, r)
    albedo = _albedo_for(style, size, r)
    lit = _shade(mask, albedo, bulge=r * 0.92, gloss=style.gloss)

    # Contact shadow inside the lower-right edge grounds the shape.
    edge = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge = np.clip(edge / max(1.0, r * 0.30), 0.0, 1.0)[..., None]
    lit *= 0.62 + 0.38 * edge

    rgb = lit.astype(np.uint8)
    if style.stem:
        _draw_stem(rgb, size, r, style.leaf)
        stem_mask = np.zeros((size, size), np.uint8)
        _draw_stem(stem_mask, size, r, style.leaf)
        mask = np.maximum(mask, (stem_mask > 0).astype(np.uint8) * 255)

    return _finish(rgb, mask, out_size)


@lru_cache(maxsize=64)
def bomb_sprite(radius: int) -> Sprite:
    """Bake the bomb body (fuse spark is animated at draw time)."""
    s = SUPERSAMPLE
    out_size = int(radius * 2 * 1.42)
    size = out_size * s
    r = radius * s
    c = size // 2

    mask = _sphere_mask(size, r)
    albedo = np.zeros((size, size, 3), np.float32)
    albedo[:] = (34, 32, 38)
    lit = _shade(mask, albedo, bulge=r * 0.92, gloss=1.0, ambient=0.30, rim=0.85)
    rgb = lit.astype(np.uint8)

    # Machined band + warning chevrons read instantly as "do not touch".
    cv2.ellipse(
        rgb, (c, c), (int(r * 0.99), int(r * 0.30)), 12, 0, 360, (58, 56, 66), int(r * 0.10), cv2.LINE_AA
    )
    for k in range(-2, 3):
        cv2.ellipse(
            rgb,
            (c + int(k * r * 0.42), c + int(k * r * 0.09)),
            (int(r * 0.09), int(r * 0.16)),
            12,
            0,
            360,
            DANGER,
            -1,
            cv2.LINE_AA,
        )

    # Screw cap the fuse comes out of.
    cap = (c - int(r * 0.10), c - int(r * 0.86))
    cv2.ellipse(rgb, cap, (int(r * 0.26), int(r * 0.16)), -18, 0, 360, (84, 82, 92), -1, cv2.LINE_AA)
    cv2.circle(mask, cap, int(r * 0.24), 255, -1, cv2.LINE_AA)
    return _finish(rgb, mask, out_size)


@lru_cache(maxsize=512)
def cut_face(name: str, radius: int, skin: BGR | None = None) -> np.ndarray:
    """Bake the wet inner face revealed by a slice. Returns BGRA uint8."""
    style = style_for(name, skin)
    s = SUPERSAMPLE
    out_size = int(radius * 2 * 1.05)
    size = out_size * s
    r = radius * s
    c = size // 2

    rgb = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (c, c), int(r), 255, -1, cv2.LINE_AA)

    cv2.circle(rgb, (c, c), int(r), style.rind or style.skin_shadow, -1, cv2.LINE_AA)
    cv2.circle(rgb, (c, c), int(r * 0.90), style.flesh, -1, cv2.LINE_AA)
    if style.core is not None:
        cv2.circle(rgb, (c, c), int(r * 0.74), style.core, -1, cv2.LINE_AA)

    if style.segments:
        for i in range(style.segments):
            a = 2 * math.pi * i / style.segments
            cv2.line(
                rgb,
                (c, c),
                (int(c + r * 0.86 * math.cos(a)), int(c + r * 0.86 * math.sin(a))),
                _mix(style.flesh, (255, 255, 255), 0.55),
                max(1, int(r * 0.045)),
                cv2.LINE_AA,
            )
        cv2.circle(rgb, (c, c), int(r * 0.10), _mix(style.flesh, (255, 255, 255), 0.6), -1, cv2.LINE_AA)

    if style.seeds:
        rng = np.random.default_rng(11)
        for _ in range(7):
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0.25, 0.62) * r
            cv2.ellipse(
                rgb,
                (int(c + d * math.cos(a)), int(c + d * math.sin(a))),
                (max(1, int(r * 0.055)), max(2, int(r * 0.085))),
                float(math.degrees(a)),
                0,
                360,
                (28, 26, 34),
                -1,
                cv2.LINE_AA,
            )

    # Wet sheen across the face.
    sheen = np.zeros((size, size, 3), np.float32)
    cv2.ellipse(
        sheen,
        (c - int(r * 0.24), c - int(r * 0.30)),
        (int(r * 0.52), int(r * 0.30)),
        -32,
        0,
        360,
        (34.0, 34.0, 34.0),
        -1,
        cv2.LINE_AA,
    )
    sheen = cv2.GaussianBlur(sheen, (0, 0), max(1.0, r * 0.10))
    rgb = np.clip(rgb.astype(np.float32) + sheen, 0, 255).astype(np.uint8)

    return _finish(rgb, mask, out_size).bgra


@dataclass
class _Cache:
    warmed: bool = False
    names: tuple[str, ...] = field(default_factory=tuple)


_warm = _Cache()


def warm_sprites(names: tuple[str, ...], radii: range) -> None:
    """Pre-bake sprites so the first spawns don't stutter."""
    if _warm.warmed and _warm.names == names:
        return
    for radius in radii:
        for name in names:
            fruit_sprite(name, radius)
        bomb_sprite(radius)
    _warm.warmed = True
    _warm.names = names
