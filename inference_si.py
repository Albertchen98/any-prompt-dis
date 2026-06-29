import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flowdis.sampling import flowdis_predict
from flowdis.util import green_screen, load_models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowDIS single-image inference")
    parser.add_argument(
        "--root-model-dir",
        type=Path,
        default=None,
        help="Root model directory. If omitted, weights are downloaded from PAIR/FlowDIS.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for the target foreground. Use an empty string for unguided DIS.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output mask PNG path.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square inference resolution.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2,
        help="Number of flow-matching sampling steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--greenscreen-path",
        type=Path,
        default=None,
        help="Optional green-screen composite output path.",
    )
    parser.add_argument(
        "--cutout-path",
        type=Path,
        default=None,
        help="Optional RGBA cutout output path.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Alpha threshold for --cutout-path.",
    )
    return parser.parse_args()


def save_greenscreen(image: Image.Image, mask: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    composite = green_screen(np.array(image.convert("RGB")), np.array(mask.convert("L")))
    Image.fromarray(composite).save(path)


def save_cutout(image: Image.Image, mask: Image.Image, path: Path, threshold: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))
    alpha = np.where(mask_np > threshold, 255, 0).astype(np.uint8)
    rgba = np.dstack([image_np, alpha])
    Image.fromarray(rgba, mode="RGBA").save(path)


def main() -> int:
    args = get_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models on %s.", args.device)
    models = load_models(root_model_dir=args.root_model_dir, device=args.device)

    logger.info("Reading %s.", args.image_path)
    image = Image.open(args.image_path).convert("RGB")

    logger.info(
        "Running FlowDIS: prompt=%r resolution=%d num_steps=%d.",
        args.prompt,
        args.resolution,
        args.num_steps,
    )
    pred_mask = flowdis_predict(
        image=image,
        prompt=args.prompt,
        models=models,
        resolution=args.resolution,
        num_inference_steps=args.num_steps,
        device=args.device,
    )
    pred_mask.save(args.output_path)
    logger.info("Saved mask to %s.", args.output_path)

    if args.greenscreen_path is not None:
        save_greenscreen(image, pred_mask, args.greenscreen_path)
        logger.info("Saved green-screen composite to %s.", args.greenscreen_path)

    if args.cutout_path is not None:
        save_cutout(image, pred_mask, args.cutout_path, args.threshold)
        logger.info("Saved RGBA cutout to %s.", args.cutout_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
