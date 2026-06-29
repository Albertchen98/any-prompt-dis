"""Visualization helpers: click markers, debug overlays, and composites.

Reuses flowdis.util.green_screen for the green-screen variant and the demo/app.py
transparent-PNG recipe for the RGBA composite.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from flowdis.util import green_screen


def marker_radius(image_size: tuple[int, int]) -> int:
    """Scale the click-dot radius to the image so it is visible but not occluding."""
    w, h = image_size
    return max(6, int(round(0.01 * min(w, h))))


def draw_marker(image: Image.Image, point: tuple[int, int], radius: int | None = None) -> Image.Image:
    """Return an RGB copy of `image` with a red dot (white outline) drawn at `point`.

    The original image is never mutated; the marked copy goes only to the VLM.
    """
    img = image.convert("RGB").copy()
    r = radius if radius is not None else marker_radius(img.size)
    x, y = int(point[0]), int(point[1])
    draw = ImageDraw.Draw(img)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255), width=max(2, r // 3))
    return img


def draw_debug(
    image: Image.Image,
    bbox_raw: tuple[int, int, int, int] | None = None,
    bbox_padded: tuple[int, int, int, int] | None = None,
    point: tuple[int, int] | None = None,
    label: str = "",
) -> Image.Image:
    """Overlay grounded bbox (raw + padded) and click point on a copy of the image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    lw = max(2, int(round(0.003 * min(img.size))))
    if bbox_padded is not None:
        draw.rectangle(list(bbox_padded), outline=(0, 180, 255), width=lw)  # cyan = padded crop
    if bbox_raw is not None:
        draw.rectangle(list(bbox_raw), outline=(0, 255, 0), width=lw)       # green = raw grounding
    if point is not None:
        r = marker_radius(img.size)
        x, y = int(point[0]), int(point[1])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    if label:
        anchor = bbox_raw or bbox_padded
        ty = max(0, (anchor[1] - 2 * lw) if anchor else 4)
        tx = (anchor[0] if anchor else 4)
        draw.text((tx + 3, ty), label, fill=(0, 255, 0))
    return img


def binarize(mask: Image.Image, thresh: int = 127) -> np.ndarray:
    """Soft FlowDIS mask -> crisp 0/255 alpha.

    FlowDIS returns a soft, sometimes low-contrast mask (e.g. max ~210). Using it directly
    as alpha makes the cutout a faint ghost, so we threshold it for a solid object. The
    threshold scales to the mask's own max so a weak-but-present object still survives.
    """
    m = np.array(mask.convert("L"))
    hi = int(m.max())
    t = thresh if hi >= 200 else max(40, int(hi * 0.5))  # adapt to weak masks
    return np.where(m >= t, 255, 0).astype(np.uint8)


def to_transparent_png(image: Image.Image, mask: Image.Image) -> Image.Image:
    """RGBA composite: foreground kept, background transparent (demo/app.py recipe)."""
    img_np = np.array(image.convert("RGB"))
    alpha = binarize(mask)
    blacked = (img_np * (alpha[:, :, None] > 0).astype(np.uint8)).astype(np.uint8)
    return Image.fromarray(np.dstack([blacked, alpha]))


def _checkerboard(size: tuple[int, int], cell: int = 24) -> np.ndarray:
    """RGB checkerboard, so 'transparent' regions read clearly in an opaque viewer."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    tile = (((xx // cell) + (yy // cell)) % 2).astype(np.uint8)
    light, dark = 235, 200
    base = np.where(tile == 0, light, dark).astype(np.uint8)
    return np.dstack([base, base, base])


def to_rgb_preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Opaque RGB cutout: foreground over a checkerboard.

    The ImageSlider renders RGB reliably (mixing an RGB original with an RGBA cutout can
    fail to display), so this is what we feed the slider; the RGBA PNG is for download.
    """
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    a = (binarize(mask)[:, :, None] / 255.0)
    bg = _checkerboard(image.size).astype(np.float32)
    out = img_np * a + bg * (1 - a)
    return Image.fromarray(out.astype(np.uint8))


def to_mask_overlay(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (0, 220, 120),
    opacity: float = 0.55,
) -> Image.Image:
    """Same-size RGB overlay: tint the predicted mask over the original image."""
    img = image.convert("RGB")
    if mask.size != img.size:
        mask = mask.resize(img.size)
    img_np = np.array(img).astype(np.float32)
    alpha = (binarize(mask)[:, :, None] / 255.0) * float(opacity)
    tint = np.zeros_like(img_np)
    tint[:, :] = color
    out = img_np * (1 - alpha) + tint * alpha
    return Image.fromarray(out.astype(np.uint8))


def to_green_screen(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Green-screen composite via flowdis.util.green_screen."""
    return Image.fromarray(green_screen(np.array(image.convert("RGB")), np.array(mask.convert("L"))))
