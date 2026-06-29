"""CLI harness: VLM-grounded crop-then-segment over a directory of images.

Spec JSON maps each image to either a complex text prompt (Application A) or a
clicked pixel (Application B):

    {
        "0.jpg": {"text": "the cup on the table, NOT the one on the stove"},
        "3.jpg": {"point": [512, 380]}
    }

Stages (single 96GB GPU cannot hold VLM + FlowDIS together):
    --stage ground    load VLM, write grounding.json, exit
    --stage segment   read grounding.json, run FlowDIS, write masks/composites
    --stage all       run ground in a child process, then segment in the parent
    --calibrate       ground + draw bbox overlays only (validate coords first)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_agent_seg")


def get_args():
    p = argparse.ArgumentParser(description="VLM-grounded crop-then-segment for FlowDIS")
    p.add_argument("--spec", type=Path, required=True, help="JSON: {image: {text|point}}")
    p.add_argument("--examples-dir", type=Path, default=Path("assets/examples"))
    p.add_argument("--flowdis-dir", type=str, default=None,
                   help="FlowDIS weights dir. Omit to auto-download from HF (PAIR/FlowDIS).")
    p.add_argument("--vlm-path", type=str, default=os.environ.get("QWEN_VLM_PATH", ""),
                   help="Local Qwen-VL weights dir (or set the QWEN_VLM_PATH env var). "
                        "This CLI uses the local VLM backend; the cloud backend lives in "
                        "agent/gradio_app.py and inference_grounded.py.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--pad-frac", type=float, default=0.12)
    p.add_argument("--crop-prompt", choices=["label", "empty"], default="label")
    p.add_argument("--stage", choices=["all", "ground", "segment"], default="all")
    p.add_argument("--calibrate", action="store_true",
                   help="Ground + draw bbox overlays only; no segmentation.")
    p.add_argument("--baseline", action="store_true",
                   help="Also run whole-image FlowDIS for comparison.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from agent.pipeline import (
        load_spec, run_calibrate_stage, run_ground_stage, run_segment_stage,
    )

    spec = load_spec(args.spec)

    if args.calibrate:
        run_calibrate_stage(spec, args.examples_dir, args.vlm_path,
                            args.output_dir, pad_frac=args.pad_frac, device=args.device)
        return

    grounding_path = args.output_dir / "grounding.json"

    if args.stage in ("all", "ground"):
        if args.stage == "all":
            # Run grounding in a child process so VLM VRAM is fully reclaimed on exit.
            cmd = [sys.executable, __file__, "--stage", "ground",
                   "--spec", str(args.spec), "--examples-dir", str(args.examples_dir),
                   "--vlm-path", args.vlm_path, "--output-dir", str(args.output_dir),
                   "--device", args.device]
            logger.info("launching grounding child: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            run_ground_stage(spec, args.examples_dir, args.vlm_path,
                             args.output_dir, device=args.device)
            return

    if args.stage in ("all", "segment"):
        if not grounding_path.exists():
            raise FileNotFoundError(f"missing {grounding_path}; run --stage ground first")
        with open(grounding_path) as f:
            grounded = json.load(f)
        run_segment_stage(
            grounded, args.examples_dir, args.flowdis_dir, args.output_dir,
            resolution=args.resolution, num_steps=args.num_steps,
            pad_frac=args.pad_frac, crop_prompt=args.crop_prompt,
            baseline=args.baseline, device=args.device,
        )


if __name__ == "__main__":
    main()
