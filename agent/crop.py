"""Crop padding/clamping and the crop -> FlowDIS -> paste-back operation."""

from __future__ import annotations

from PIL import Image

from flowdis.sampling import flowdis_predict


def pad_and_clamp(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    pad_frac: float = 0.12,
    pad_abs_min: int = 8,
) -> tuple[int, int, int, int]:
    """Expand bbox by pad_frac of its own size (context for FlowDIS), clamp to image.

    A small pad gives FlowDIS clean boundaries without pulling the excluded neighbor
    back into the crop. Raises ValueError on a degenerate box.
    """
    x1, y1, x2, y2 = bbox
    W, H = image_size
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate bbox {bbox}")
    if pad_frac <= 0:
        px = py = 0
    else:
        px = max(pad_abs_min, round(bw * pad_frac))
        py = max(pad_abs_min, round(bh * pad_frac))
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(W, x2 + px),
        min(H, y2 + py),
    )


def crop_predict_paste(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    models,
    label: str,
    resolution: int = 1024,
    num_inference_steps: int = 2,
    device: str = "cuda",
) -> tuple[Image.Image, Image.Image]:
    """Crop `bbox`, run FlowDIS on the crop, paste its mask into a full-image mask.

    Returns (full_mask, crop_mask). full_mask is an "L" image at the original size.
    flowdis_predict already returns a mask at the crop's size, so the paste is exact.
    """
    crop = image.crop(bbox)
    crop_mask = flowdis_predict(
        image=crop,
        prompt=label,
        models=models,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    full_mask = Image.new("L", image.size, 0)
    # defensive resize covers any rounding between crop size and mask size
    full_mask.paste(crop_mask.resize((bbox[2] - bbox[0], bbox[3] - bbox[1])), (bbox[0], bbox[1]))
    return full_mask, crop_mask
