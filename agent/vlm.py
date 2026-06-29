"""Qwen3.5-VL wrapper for object grounding (text disambiguation + point click).

Loads the local qwen3.5-27b checkpoint via the Auto* classes (transformers >= 5.x,
which recognizes the `qwen3_5` architecture). Generalizes the messages/processor/
generate pattern from demo/qwen.py and adds grounding + coordinate conversion.
"""

from __future__ import annotations

import gc
import logging
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from agent.grounding import (
    POINT_GROUNDING_PROMPT,
    TEXT_GROUNDING_PROMPT,
    GroundedObject,
    model_bbox_to_orig,
    parse_grounding,
)
from agent.viz import draw_marker

logger = logging.getLogger(__name__)

# Local (offline) Qwen-VL weights. Set via the QWEN_VLM_PATH env var or pass model_path=...
# The default backend is the cloud VLM (agent.cloud_vlm.CloudVLM), which needs no local
# weights; this local backend is the optional offline alternative.
DEFAULT_VLM_PATH = os.environ.get("QWEN_VLM_PATH", "")


class VLM:
    """Vision-language grounding model wrapper."""

    def __init__(self, model_path: str = DEFAULT_VLM_PATH, device: str = "cuda"):
        if not model_path:
            raise ValueError(
                "Local VLM weights path is empty. Set the QWEN_VLM_PATH env var or pass "
                "model_path=... . For the default cloud backend use "
                "agent.cloud_vlm.CloudVLM instead (no local weights needed)."
            )
        logger.info("Loading VLM from %s", model_path)
        self.model_path = model_path
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.patch_size = int(self.processor.image_processor.patch_size)
        logger.info("VLM loaded (patch_size=%d).", self.patch_size)

    def free(self) -> None:
        """Release the model from GPU so FlowDIS can be loaded afterwards."""
        try:
            self.model = None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "VLM freed. cuda allocated=%.2f GB reserved=%.2f GB",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0,
            )

    @torch.no_grad()
    def _run(self, image: Image.Image, prompt: str, max_new_tokens: int = 1280):
        """Run one VLM turn. Returns (decoded_text, image_grid_thw_list)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        grid_thw = inputs["image_grid_thw"][0].tolist()
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return out, grid_thw

    def ground_from_text(self, image: Image.Image, user_prompt: str) -> GroundedObject:
        """Application A: reason over a complex prompt, ground the chosen object."""
        prompt = TEXT_GROUNDING_PROMPT.format(user_prompt=user_prompt)
        raw, grid_thw = self._run(image, prompt)
        parsed = parse_grounding(raw)
        bbox, hyp = model_bbox_to_orig(
            parsed["bbox_2d"], grid_thw, self.patch_size, image.size
        )
        logger.info("ground_from_text label=%r bbox=%s hyp=%s", parsed["label"], bbox, hyp)
        return GroundedObject(
            label=parsed["label"],
            bbox=bbox,
            source="text",
            input=user_prompt,
            raw=raw,
            coord_hypothesis=hyp,
            bbox_model=tuple(int(round(v)) for v in parsed["bbox_2d"]),
        )

    def ground_from_point(self, image: Image.Image, point: tuple[int, int]) -> GroundedObject:
        """Application B: confirm the object under a clicked pixel, ground it.

        Draws a red dot on a copy passed to the VLM (image-space cue) and also passes
        the raw coordinate in text as a backup cue.
        """
        marked = draw_marker(image, point)
        raw, grid_thw = self._run(marked, POINT_GROUNDING_PROMPT)
        parsed = parse_grounding(raw)
        bbox, hyp = model_bbox_to_orig(
            parsed["bbox_2d"], grid_thw, self.patch_size, image.size
        )
        x, y = int(point[0]), int(point[1])
        contains = bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
        if not contains:
            logger.warning(
                "grounded bbox %s does NOT contain click point %s (label=%r)",
                bbox, point, parsed["label"],
            )
        logger.info("ground_from_point label=%r bbox=%s hyp=%s contains=%s",
                    parsed["label"], bbox, hyp, contains)
        return GroundedObject(
            label=parsed["label"],
            bbox=bbox,
            source="point",
            input=f"[{x}, {y}]",
            raw=raw,
            coord_hypothesis=hyp,
            bbox_model=tuple(int(round(v)) for v in parsed["bbox_2d"]),
        )
