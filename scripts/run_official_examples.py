#!/usr/bin/env python3
"""Run quantized FlowDIS on every official example image.

The image name, prompt, inference resolution, and sampling step count are read
from ``assets/examples/examples.csv``.  The INT8 ConvRot transformer is loaded
once and reused for the complete batch.

Example:
    python scripts/run_official_examples.py \
        --root-model-dir /mnt/data1/weights/FlowDIS \
        --output-dir official_examples_int8_out

Add ``--t5-int4`` to test the full low-VRAM configuration (INT8 DiT + INT4
T5).  Without it, only the DiT is quantized and the regular bf16 T5 is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowdis.sampling import flowdis_predict  # noqa: E402
from flowdis.util import green_screen, load_models  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_official_examples")

DEFAULT_EXAMPLES_DIR = REPO_ROOT / "assets" / "examples"


@dataclass(frozen=True)
class Example:
    image_name: str
    prompt: str
    resolution: int
    num_steps: int


@dataclass(frozen=True)
class Result:
    image_name: str
    prompt: str
    resolution: int
    num_steps: int
    elapsed_seconds: float
    foreground_fraction: float
    mask_min: int
    mask_max: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run INT8 ConvRot FlowDIS on all official example images."
    )
    parser.add_argument(
        "--root-model-dir",
        type=Path,
        default=Path(os.environ["FLOWDIS_DIR"]) if os.environ.get("FLOWDIS_DIR") else None,
        help=(
            "Directory containing the base FlowDIS files and "
            "flowdis-transformer-int8-convrot.safetensors. Defaults to FLOWDIS_DIR."
        ),
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=DEFAULT_EXAMPLES_DIR,
        help="Directory containing examples.csv and the official input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "official_examples_int8_out",
        help="Destination for masks, green-screen previews, and summary.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch inference device, for example cuda or cuda:1.",
    )
    parser.add_argument(
        "--t5-int4",
        action="store_true",
        help="Also use the nunchaku AWQ-INT4 T5 encoder.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Mask threshold used for the foreground fraction in summary.json.",
    )
    args = parser.parse_args()

    if args.root_model_dir is None:
        parser.error("--root-model-dir is required (or set FLOWDIS_DIR)")
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested, but torch.cuda.is_available() is False")
    return args


def load_manifest(examples_dir: Path) -> list[Example]:
    manifest_path = examples_dir / "examples.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Official example manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected_columns = {"image_name", "prompt", "resolution", "num_steps"}
        missing_columns = expected_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{manifest_path} is missing columns: {sorted(missing_columns)}"
            )

        examples = [
            Example(
                image_name=row["image_name"].strip(),
                prompt=row["prompt"],
                resolution=int(row["resolution"]),
                num_steps=int(row["num_steps"]),
            )
            for row in reader
        ]

    if not examples:
        raise ValueError(f"No examples listed in {manifest_path}")

    names = [example.image_name for example in examples]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate image names in {manifest_path}: {duplicates}")

    missing_images = [name for name in names if not (examples_dir / name).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Images listed in the manifest are missing: {missing_images}")

    image_suffixes = {".jpg", ".jpeg", ".png"}
    available_images = {
        path.name for path in examples_dir.iterdir() if path.suffix.lower() in image_suffixes
    }
    unlisted_images = sorted(available_images.difference(names))
    if unlisted_images:
        raise ValueError(
            f"Images in {examples_dir} are not listed in examples.csv: {unlisted_images}"
        )

    return examples


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def main() -> int:
    args = parse_args()
    examples = load_manifest(args.examples_dir)

    int8_checkpoint = args.root_model_dir / "flowdis-transformer-int8-convrot.safetensors"
    if not int8_checkpoint.is_file():
        raise FileNotFoundError(f"INT8 ConvRot checkpoint not found: {int8_checkpoint}")
    if args.t5_int4:
        t5_checkpoint = (
            args.root_model_dir
            / "nunchaku-t5"
            / "awq-int4-flux.1-t5xxl.safetensors"
        )
        if not t5_checkpoint.is_file():
            raise FileNotFoundError(f"INT4 T5 checkpoint not found: {t5_checkpoint}")

    masks_dir = args.output_dir / "masks"
    previews_dir = args.output_dir / "greenscreens"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading quantized FlowDIS once on %s (INT8 DiT, T5=%s).",
        args.device,
        "INT4" if args.t5_int4 else "bf16",
    )
    models = load_models(
        root_model_dir=args.root_model_dir,
        device=args.device,
        int8=True,
        t5_int4=args.t5_int4,
    )

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    results: list[Result] = []
    for index, example in enumerate(examples, start=1):
        image_path = args.examples_dir / example.image_name
        logger.info(
            "[%d/%d] %s | prompt=%r | resolution=%d | steps=%d",
            index,
            len(examples),
            example.image_name,
            example.prompt,
            example.resolution,
            example.num_steps,
        )

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        synchronize(args.device)
        started_at = time.perf_counter()
        mask = flowdis_predict(
            image=image,
            prompt=example.prompt,
            models=models,
            resolution=example.resolution,
            num_inference_steps=example.num_steps,
            device=args.device,
        )
        synchronize(args.device)
        elapsed = time.perf_counter() - started_at

        stem = Path(example.image_name).stem
        mask_path = masks_dir / f"{stem}.png"
        preview_path = previews_dir / f"{stem}.png"
        mask.save(mask_path)

        mask_array = np.asarray(mask.convert("L"))
        preview = green_screen(np.asarray(image), mask_array)
        Image.fromarray(preview).save(preview_path)

        result = Result(
            image_name=example.image_name,
            prompt=example.prompt,
            resolution=example.resolution,
            num_steps=example.num_steps,
            elapsed_seconds=round(elapsed, 3),
            foreground_fraction=round(float((mask_array > args.threshold).mean()), 6),
            mask_min=int(mask_array.min()),
            mask_max=int(mask_array.max()),
        )
        results.append(result)
        logger.info(
            "Saved %s and %s (%.2fs, foreground %.1f%%).",
            mask_path,
            preview_path,
            elapsed,
            100.0 * result.foreground_fraction,
        )

    summary: dict[str, object] = {
        "configuration": {
            "root_model_dir": str(args.root_model_dir),
            "examples_dir": str(args.examples_dir),
            "device": args.device,
            "transformer": "int8-convrot",
            "t5": "int4" if args.t5_int4 else "bf16",
            "threshold": args.threshold,
        },
        "examples": [asdict(result) for result in results],
        "total_elapsed_seconds": round(sum(r.elapsed_seconds for r in results), 3),
    }
    if args.device.startswith("cuda"):
        summary["peak_cuda_memory_gib"] = round(
            torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**3,
            3,
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Finished %d examples. Summary: %s", len(results), summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
