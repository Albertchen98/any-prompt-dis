"""Single-image VLM-grounded segmentation: complex text -> bbox + object prompt -> crop -> FlowDIS.

Given ONE image and ONE complex text description, a cloud VLM (OpenAI-compatible or Gemini
API) reasons about which object the user means, grounds it to a bounding box, and names it.
We crop that region, run FlowDIS on the clean crop with the object prompt, and paste the
mask back into a full-image mask.

The VLM runs in the cloud, so it uses no GPU/VRAM and FlowDIS stays resident: the whole
pipeline runs in this one process. Example:

    python inference_grounded.py \
        --image-path assets/examples/1.jpg \
        --prompt "the left tower of the bridge, not the right one" \
        --output-path out/1_mask.png --composite-path out/1_cutout.png \
        --grounding-result-path out/1_grounding_result.png

Set VLM_API_KEY (and optionally VLM_API_FORMAT=openai|gemini, VLM_API_BASE, VLM_PROXY).
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from PIL import Image

from agent.cloud_vlm import (
    DEFAULT_API_BASE,
    DEFAULT_API_FORMAT,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_SIDE,
    DEFAULT_MODEL,
    DEFAULT_PROXY,
    CloudVLM,
)
from agent.pipeline import segment_grounded
from agent.viz import draw_grounding_result, to_green_screen, to_mask_overlay, to_transparent_png
from flowdis.util import load_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_grounded")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-image VLM-grounded crop-then-segment for FlowDIS"
    )
    p.add_argument("--image-path", type=Path, required=True, help="Input image path.")
    p.add_argument(
        "--prompt", type=str, required=True,
        help="Complex text description of the target object "
             '(e.g. "the cup on the table, NOT the one on the stove").',
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output mask PNG path. Required unless --grounding-only is set.",
    )
    p.add_argument(
        "--grounding-path",
        type=Path,
        default=None,
        help="Optional JSON path for the VLM grounding result.",
    )
    p.add_argument(
        "--grounding-only",
        action="store_true",
        help="Only run VLM grounding and bbox visualization; skip FlowDIS segmentation.",
    )
    p.add_argument(
        "--composite-path", type=Path, default=None,
        help="Optional RGBA cutout (transparent background) output path.",
    )
    p.add_argument(
        "--greenscreen-path", type=Path, default=None,
        help="Optional green-screen composite output path.",
    )
    p.add_argument(
        "--overlay-path",
        type=Path,
        default=None,
        help="Optional original-image preview with the mask shown as a random translucent color.",
    )
    p.add_argument(
        "--overlay-opacity",
        type=float,
        default=0.55,
        help="Opacity for --overlay-path mask tint.",
    )
    p.add_argument(
        "--grounding-result-path",
        type=Path,
        default=None,
        help="Optional original-image bbox/label overlay saved right after VLM grounding.",
    )
    p.add_argument(
        "--root-model-dir", type=Path, default=None,
        help="FlowDIS root model directory. If omitted, weights download from PAIR/FlowDIS.",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Cloud grounding model id (or set VLM_MODEL).",
    )
    p.add_argument(
        "--api-format", type=str, default=DEFAULT_API_FORMAT, choices=["openai", "gemini"],
        help="Cloud API request format (or set VLM_API_FORMAT).",
    )
    p.add_argument(
        "--api-base", type=str, default=DEFAULT_API_BASE,
        help="Cloud API base URL (or set VLM_API_BASE). Defaults per --api-format.",
    )
    p.add_argument("--resolution", type=int, default=1024, help="Square inference resolution.")
    p.add_argument("--num-steps", type=int, default=2, help="Flow-matching sampling steps.")
    p.add_argument(
        "--pad-frac", type=float, default=0.12,
        help="Fraction of the bbox size to pad the crop by, for clean context.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device for FlowDIS.")
    p.add_argument(
        "--proxy", type=str, default=DEFAULT_PROXY,
        help="HTTP(S) proxy for the cloud API call (or set VLM_PROXY). Pass '' to disable.",
    )
    p.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                   help="Max image side uploaded to the VLM (coords are normalized).")
    p.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES,
                   help="Max VLM upload data-URL size in bytes (proxy-friendliness).")
    args = p.parse_args()
    if not args.grounding_only and args.output_path is None:
        p.error("--output-path is required unless --grounding-only is set.")
    return args


def save_grounding_outputs(
    image: Image.Image,
    grounded,
    *,
    grounding_path: Path | None = None,
    grounding_result_path: Path | None = None,
    bbox_padded: tuple[int, int, int, int] | None = None,
) -> None:
    """Save structured grounding and an original-image bbox overlay."""
    if grounding_path is not None:
        grounding_path.parent.mkdir(parents=True, exist_ok=True)
        grounding_path.write_text(
            json.dumps(grounded.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )
        logger.info("Saved grounding JSON -> %s.", grounding_path)

    if grounding_result_path is not None:
        grounding_result_path.parent.mkdir(parents=True, exist_ok=True)
        draw_grounding_result(
            image,
            bbox_raw=grounded.bbox,
            bbox_padded=bbox_padded,
            label=grounded.label,
        ).save(grounding_result_path)
        logger.info("Saved grounding result -> %s.", grounding_result_path)


def main() -> int:
    args = get_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    image = Image.open(args.image_path).convert("RGB")
    logger.info("Loaded %s (size=%s).", args.image_path, image.size)

    vlm = CloudVLM(
        model=args.model,
        proxy=(args.proxy or None),
        max_side=args.max_side,
        max_image_bytes=args.max_image_bytes,
        api_format=args.api_format,
        api_base=args.api_base,
    )

    logger.info("Grounding: prompt=%r", args.prompt)
    grounded = vlm.ground_from_text(image, args.prompt, model=args.model)

    logger.info(
        "Grounded object_prompt=%r bbox=%s (coord=%s).",
        grounded.label, grounded.bbox, grounded.coord_hypothesis,
    )

    save_grounding_outputs(
        image,
        grounded,
        grounding_path=args.grounding_path,
        grounding_result_path=args.grounding_result_path,
    )

    if args.grounding_only:
        print(json.dumps(grounded.to_dict(), ensure_ascii=False))
        return 0

    logger.info("Loading FlowDIS on %s.", args.device)
    models = load_models(root_model_dir=args.root_model_dir, device=args.device)

    logger.info("Segmenting grounded crop.")
    full_mask, bbox_pad = segment_grounded(
        image, grounded, models,
        resolution=args.resolution, num_steps=args.num_steps,
        pad_frac=args.pad_frac, device=args.device,
    )

    logger.info(
        "Grounded object_prompt=%r bbox=%s (padded crop=%s, coord=%s).",
        grounded.label, grounded.bbox, bbox_pad, grounded.coord_hypothesis,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    full_mask.save(args.output_path)
    logger.info("Saved mask -> %s.", args.output_path)

    if args.composite_path is not None:
        args.composite_path.parent.mkdir(parents=True, exist_ok=True)
        to_transparent_png(image, full_mask).save(args.composite_path)
        logger.info("Saved RGBA cutout -> %s.", args.composite_path)

    if args.greenscreen_path is not None:
        args.greenscreen_path.parent.mkdir(parents=True, exist_ok=True)
        to_green_screen(image, full_mask).save(args.greenscreen_path)
        logger.info("Saved green-screen composite -> %s.", args.greenscreen_path)

    if args.overlay_path is not None:
        rng = random.SystemRandom()
        color = tuple(rng.randint(64, 255) for _ in range(3))
        args.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        to_mask_overlay(
            image,
            full_mask,
            color=color,
            opacity=args.overlay_opacity,
        ).save(args.overlay_path)
        logger.info(
            "Saved mask overlay -> %s (color=%s, opacity=%.2f).",
            args.overlay_path,
            color,
            args.overlay_opacity,
        )

    # Final machine-readable summary line: the bbox and object prompt the user asked for.
    print(json.dumps(grounded.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
