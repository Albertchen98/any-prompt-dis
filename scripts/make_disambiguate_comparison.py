"""Build the README visualization for the pedal-kart disambiguation example.

This script does not run inference. It takes two already predicted full-resolution
masks, applies an identical overlay, and crops the same display-only region from the
input and both results so that the small target is legible in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.viz import binarize  # noqa: E402


DEFAULT_ROI = (590, 380, 1000, 620)
OVERLAY_COLOR = np.array((0, 190, 255), dtype=np.float32)
OUTLINE_COLOR = np.array((255, 230, 0), dtype=np.uint8)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full-scene and zoomed FlowDIS disambiguation comparisons."
    )
    parser.add_argument(
        "--image", type=Path, default=REPO_ROOT / "assets" / "Playarena-Indoor.jpg"
    )
    parser.add_argument(
        "--plain-mask",
        type=Path,
        default=REPO_ROOT / "assets" / "disambiguate" / "plain_flowdis_mask.png",
    )
    parser.add_argument(
        "--disambiguated-mask",
        type=Path,
        default=REPO_ROOT / "assets" / "disambiguate" / "mask.png",
    )
    parser.add_argument(
        "--grounding-json",
        type=Path,
        default=REPO_ROOT / "assets" / "disambiguate" / "grounding.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "disambiguate" / "comparison",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_ROI,
        help="Display-only crop containing both ambiguous pedal karts.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=2,
        help="Integer scale factor for the three display-only crops.",
    )
    return parser.parse_args()


def validate_box(box: tuple[int, int, int, int], size: tuple[int, int], name: str) -> None:
    x1, y1, x2, y2 = box
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name} {box} is outside image size {size}.")


def mask_overlay(image: Image.Image, mask: Image.Image, opacity: float = 0.48) -> Image.Image:
    """Apply a deterministic cyan fill and yellow boundary to a full-size mask."""
    if mask.size != image.size:
        raise ValueError(f"Mask size {mask.size} does not match image size {image.size}.")

    foreground = binarize(mask) > 0
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    output = image_array.copy()
    output[foreground] = (
        image_array[foreground] * (1.0 - opacity) + OVERLAY_COLOR * opacity
    )

    binary_image = Image.fromarray((foreground * 255).astype(np.uint8), mode="L")
    dilated = np.asarray(binary_image.filter(ImageFilter.MaxFilter(3))) > 0
    eroded = np.asarray(binary_image.filter(ImageFilter.MinFilter(3))) > 0
    output[dilated ^ eroded] = OUTLINE_COLOR
    return Image.fromarray(output.astype(np.uint8), mode="RGB")


def zoom_crop(image: Image.Image, roi: tuple[int, int, int, int], zoom: int) -> Image.Image:
    crop = image.crop(roi)
    return crop.resize((crop.width * zoom, crop.height * zoom), Image.Resampling.LANCZOS)


def make_overview(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    grounding_box: tuple[int, int, int, int],
) -> Image.Image:
    overview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overview)
    line_width = max(3, round(min(image.size) * 0.006))
    draw.rectangle(roi, outline=(255, 145, 0), width=line_width)
    draw.rectangle(grounding_box, outline=(0, 255, 0), width=line_width)
    return overview


def main() -> int:
    args = get_args()
    if args.zoom < 1:
        raise ValueError("--zoom must be at least 1.")

    image = Image.open(args.image).convert("RGB")
    plain_mask = Image.open(args.plain_mask).convert("L")
    disambiguated_mask = Image.open(args.disambiguated_mask).convert("L")
    roi = tuple(args.roi)
    validate_box(roi, image.size, "ROI")

    grounding = json.loads(args.grounding_json.read_text())
    grounding_box = tuple(grounding["bbox"])
    validate_box(grounding_box, image.size, "Grounding bbox")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_overview(image, roi, grounding_box).save(args.output_dir / "overview.png")
    zoom_crop(image, roi, args.zoom).save(args.output_dir / "zoom_input.png")
    zoom_crop(mask_overlay(image, plain_mask), roi, args.zoom).save(
        args.output_dir / "zoom_plain_flowdis.png"
    )
    zoom_crop(mask_overlay(image, disambiguated_mask), roi, args.zoom).save(
        args.output_dir / "zoom_disambiguated.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
