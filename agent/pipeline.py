"""Two-stage orchestration: VLM grounding stage, then FlowDIS segmentation stage.

The 27B VLM (~54GB) and FlowDIS (~35-48GB) cannot co-reside on the single 96GB GPU,
so the stages are separated. `--stage all` runs grounding in a child process (which
guarantees the OS reclaims VLM VRAM on exit) before loading FlowDIS in the parent.
The two stages communicate via grounding.json.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from PIL import Image

from agent.crop import crop_predict_paste, pad_and_clamp
from agent.grounding import GroundedObject, GroundingParseError
from agent.viz import draw_debug, to_green_screen, to_transparent_png

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_spec(spec_path: Path) -> dict:
    with open(spec_path) as f:
        return json.load(f)


def resolve_image(examples_dir: Path, name: str) -> Path:
    p = examples_dir / name
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    return p


# --- single-image convenience API ---------------------------------------------
# These two functions are the reusable core of "complex text -> grounded crop ->
# FlowDIS segment". They are shared by the single-image CLI (inference_grounded.py)
# and the Gradio app, so the crop->predict->paste contract lives in exactly one place.


def segment_grounded(
    image: Image.Image,
    grounded: GroundedObject,
    models,
    *,
    resolution: int = 1024,
    num_steps: int = 2,
    pad_frac: float = 0.12,
    device: str = "cuda",
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop the grounded bbox (padded for clean context), run FlowDIS, paste back.

    Uses the grounded object's `label` as the FlowDIS prompt: a concise object prompt
    guides figure-ground separation, whereas an empty prompt makes FlowDIS keep
    low-texture background (e.g. sky) on background-heavy crops.

    Returns (full_mask, bbox_padded). full_mask is an "L" image at the original size.
    """
    bbox_pad = pad_and_clamp(grounded.bbox, image.size, pad_frac=pad_frac)
    full_mask, _ = crop_predict_paste(
        image, bbox_pad, models, grounded.label or "",
        resolution=resolution, num_inference_steps=num_steps, device=device,
    )
    return full_mask, bbox_pad


def ground_and_segment(
    image: Image.Image,
    text: str,
    vlm,
    models,
    *,
    model: str | None = None,
    resolution: int = 1024,
    num_steps: int = 2,
    pad_frac: float = 0.12,
    device: str = "cuda",
) -> tuple[Image.Image, GroundedObject, tuple[int, int, int, int]]:
    """Complex text description -> VLM grounds the target (bbox + object prompt) ->
    crop the target region -> FlowDIS segment on the crop.

    `vlm` is any grounding backend exposing `ground_from_text` (agent.vlm.VLM or
    agent.cloud_vlm.CloudVLM). `model` selects a cloud model when the backend supports
    it (the local backend has no such argument, so it is omitted there).

    Returns (full_mask, grounded, bbox_padded). The grounded object carries the bbox
    (`grounded.bbox`, original pixels) and the object prompt (`grounded.label`).
    """
    if model is not None:
        grounded = vlm.ground_from_text(image, text, model=model)
    else:
        grounded = vlm.ground_from_text(image, text)
    full_mask, bbox_pad = segment_grounded(
        image, grounded, models,
        resolution=resolution, num_steps=num_steps, pad_frac=pad_frac, device=device,
    )
    return full_mask, grounded, bbox_pad


# --- stage 1: grounding --------------------------------------------------------


def run_ground_stage(
    spec: dict,
    examples_dir: Path,
    vlm_path: str,
    out_dir: Path,
    device: str = "cuda",
) -> dict:
    """Load the VLM, ground every spec entry, write grounding.json. Returns the dict."""
    from agent.vlm import VLM  # imported lazily so the segment stage never loads it

    out_dir.mkdir(parents=True, exist_ok=True)
    vlm = VLM(model_path=vlm_path, device=device)

    grounded: dict[str, dict] = {}
    for name, entry in spec.items():
        img_path = resolve_image(examples_dir, name)
        image = Image.open(img_path).convert("RGB")
        try:
            if "text" in entry:
                g = vlm.ground_from_text(image, entry["text"])
            elif "point" in entry:
                g = vlm.ground_from_point(image, tuple(entry["point"]))
            else:
                logger.warning("skip %s: spec entry needs 'text' or 'point'", name)
                continue
            grounded[name] = g.to_dict()
        except GroundingParseError as e:
            logger.error("grounding parse failed for %s: %s", name, e.raw)
            grounded[name] = {"error": "parse", "raw": e.raw, **entry}

    vlm.free()

    out_path = out_dir / "grounding.json"
    with open(out_path, "w") as f:
        json.dump(grounded, f, indent=2, ensure_ascii=False)
    logger.info("wrote grounding -> %s (%d entries)", out_path, len(grounded))
    return grounded


# --- stage 2: segmentation -----------------------------------------------------


def run_segment_stage(
    grounded: dict,
    examples_dir: Path,
    flowdis_dir: str,
    out_dir: Path,
    resolution: int = 1024,
    num_steps: int = 2,
    pad_frac: float = 0.12,
    crop_prompt: str = "label",
    baseline: bool = False,
    device: str = "cuda",
) -> None:
    """Load FlowDIS, crop->predict->paste for each grounded object, write artifacts."""
    from flowdis.sampling import flowdis_predict
    from flowdis.util import load_models

    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(
        root_model_dir=Path(flowdis_dir) if flowdis_dir else None, device=device
    )

    rows = []
    for name, g in grounded.items():
        if g.get("error"):
            rows.append({"image": name, "parse_ok": False, **{k: g.get(k, "") for k in
                         ("source", "input", "label", "coord_hypothesis")}})
            continue

        img_path = resolve_image(examples_dir, name)
        image = Image.open(img_path).convert("RGB")
        stem = Path(name).stem
        bbox_raw = tuple(g["bbox"])
        label = g["label"]
        # crop_prompt="empty" segments the crop with no object prompt (ablation)
        obj = GroundedObject(
            label=("" if crop_prompt == "empty" else label), bbox=bbox_raw
        )

        try:
            full_mask, bbox_pad = segment_grounded(
                image, obj, models,
                resolution=resolution, num_steps=num_steps,
                pad_frac=pad_frac, device=device,
            )
        except ValueError as e:
            logger.error("bad bbox for %s: %s", name, e)
            continue

        full_mask.save(out_dir / f"{stem}_mask.png")
        to_transparent_png(image, full_mask).save(out_dir / f"{stem}_composite.png")
        to_green_screen(image, full_mask).save(out_dir / f"{stem}_greenscreen.png")

        point = tuple(json.loads(g["input"])) if g.get("source") == "point" else None
        draw_debug(image, bbox_raw=bbox_raw, bbox_padded=bbox_pad,
                   point=point, label=label).save(out_dir / f"{stem}_debug.png")

        if baseline:
            base_prompt = g.get("input", "") if g.get("source") == "text" else label
            base_mask = flowdis_predict(
                image=image, prompt=base_prompt, models=models,
                resolution=resolution, num_inference_steps=num_steps, device=device,
            )
            base_mask.save(out_dir / f"{stem}_baseline_mask.png")
            to_transparent_png(image, base_mask).save(out_dir / f"{stem}_baseline_composite.png")

        rows.append({
            "image": name, "parse_ok": True, "source": g.get("source", ""),
            "input": g.get("input", ""), "label": label,
            "bbox": g["bbox"], "bbox_padded": list(bbox_pad),
            "coord_hypothesis": g.get("coord_hypothesis", ""),
        })
        logger.info("segmented %s (label=%r)", name, label)

    with open(out_dir / "results.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    logger.info("wrote %d results -> %s", len(rows), out_dir / "results.csv")


# --- calibration (ground only, draw overlay, no FlowDIS) -----------------------


def run_calibrate_stage(
    spec: dict,
    examples_dir: Path,
    vlm_path: str,
    out_dir: Path,
    pad_frac: float = 0.12,
    device: str = "cuda",
) -> None:
    """Ground each entry and draw a bbox overlay so the operator can eyeball the
    coordinate transform before trusting the pipeline. No segmentation."""
    grounded = run_ground_stage(spec, examples_dir, vlm_path, out_dir, device=device)
    for name, g in grounded.items():
        if g.get("error"):
            continue
        image = Image.open(resolve_image(examples_dir, name)).convert("RGB")
        bbox_raw = tuple(g["bbox"])
        bbox_pad = pad_and_clamp(bbox_raw, image.size, pad_frac=pad_frac)
        point = tuple(json.loads(g["input"])) if g.get("source") == "point" else None
        draw_debug(image, bbox_raw=bbox_raw, bbox_padded=bbox_pad,
                   point=point, label=g["label"]).save(out_dir / f"{Path(name).stem}_debug.png")
    logger.info("calibration overlays written -> %s", out_dir)
