"""Grounding contract: prompt templates, robust JSON parsing, and coordinate conversion.

Both cloud and local VLM backends are asked to emit one JSON object with a short
segmentation label plus a `bbox_2d` box. Cloud backends are explicitly requested to
return normalized 0-1000 coordinates. The local Qwen backend has historically varied
by checkpoint, so its converter tries normalized coordinates first, then resized-space
and original-pixel fallbacks for compatibility.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


# --- output contract -----------------------------------------------------------

TEXT_GROUNDING_PROMPT = (
    "You are an object-grounding assistant. Look at the image and the user request "
    "below.\n"
    "First, silently reason about which SINGLE object the user truly means, paying "
    "close attention to any negations or disambiguating conditions (for example "
    '"the one on the table, NOT the one on the stove").\n'
    "Then output ONLY a JSON object, with no other text, in exactly this format:\n"
    '{{"label": "<generic object name>", "bbox_2d": [x1, y1, x2, y2]}}\n'
    "Rules:\n"
    "- label is the SHORT, generic name of the object TYPE: a 1-3 word noun phrase. Do "
    "NOT copy the disambiguating wording from the request — no colors, owners, "
    "actions, or relative clauses (no 'ridden by ...', 'wearing ...', 'on the ...', "
    "'not the ...'). The bounding box already encodes WHICH instance; the label only "
    "names WHAT it is, as a clean segmentation prompt. Examples: 'the cup on the table, "
    "not the one on the stove' -> 'cup'; 'the pedal kart ridden by the kid in a cyan "
    "shirt' -> 'pedal kart'.\n"
    "- bbox_2d is the bounding box [left, top, right, bottom] of the chosen object, "
    "with coordinates NORMALIZED to the range 0-1000, where (0,0) is the top-left "
    "corner of the image and (1000,1000) is the bottom-right corner.\n"
    "- Choose exactly ONE object. If the request excludes an object, do NOT return it.\n"
    "- Output the JSON only. No explanation, no markdown code fences.\n\n"
    'User request: "{user_prompt}"'
)

POINT_GROUNDING_PROMPT = (
    "You are an object-grounding assistant. A red dot has been drawn on the image at "
    "a specific point. Identify the SINGLE object that the red dot is placed on top "
    "of.\n"
    "Then output ONLY a JSON object, with no other text, in exactly this format:\n"
    '{"label": "<short noun phrase for that object>", "bbox_2d": [x1, y1, x2, y2]}\n'
    "Rules:\n"
    "- bbox_2d is the bounding box [left, top, right, bottom] of the marked object, "
    "with coordinates NORMALIZED to the range 0-1000, where (0,0) is the top-left "
    "corner of the image and (1000,1000) is the bottom-right corner.\n"
    "- The bounding box must contain the red dot.\n"
    "- Judge purely from where the red dot appears in the image you see.\n"
    "- Output the JSON only. No explanation, no markdown code fences."
)
# NOTE: we deliberately do NOT pass the raw click pixel (x, y) in the text. The model
# sees a smart-resized image and reasons in a 0-1000 space, so an original-pixel
# coordinate just confuses it and wastes its token budget. The drawn dot is the cue.


TEXT_GROUNDING_PROMPT_NORM = (
    "You are an object-grounding assistant. Look at the image and the user request "
    "below.\n"
    "First, reason about which SINGLE object the user truly means, paying close "
    "attention to any negations or disambiguating conditions (for example "
    '"the one on the table, NOT the one on the stove").\n'
    "Then output ONLY a JSON object, with no other text, in exactly this format:\n"
    '{{"label": "<generic object name>", "bbox_2d": [x1, y1, x2, y2]}}\n'
    "Rules:\n"
    "- label is the SHORT, generic name of the object TYPE: a 1-3 word noun phrase. Do "
    "NOT copy the disambiguating wording from the request — no colors, owners, "
    "actions, or relative clauses (no 'ridden by ...', 'wearing ...', 'on the ...', "
    "'not the ...'). The bounding box already encodes WHICH instance; the label only "
    "names WHAT it is, as a clean segmentation prompt. Examples: 'the cup on the table, "
    "not the one on the stove' -> 'cup'; 'the pedal kart ridden by the kid in a cyan "
    "shirt' -> 'pedal kart'.\n"
    "- bbox_2d is the bounding box [left, top, right, bottom] of the chosen object, with "
    "coordinates NORMALIZED to the range 0-1000, where (0,0) is the top-left corner of "
    "the image and (1000,1000) is the bottom-right corner.\n"
    "- Make the box as TIGHT as possible around ONLY that one object. Do NOT include "
    "adjacent or nearby objects, and do NOT pad the box with extra background.\n"
    "- Choose exactly ONE object. If the request excludes an object, do NOT return it.\n"
    "- Output the JSON only. No explanation, no markdown code fences.\n\n"
    'User request: "{user_prompt}"'
)

POINT_GROUNDING_PROMPT_NORM = (
    "You are an object-grounding assistant. A red dot has been drawn on the image at a "
    "specific point. Identify the SINGLE object that the red dot is placed on top of.\n"
    "Then output ONLY a JSON object, with no other text, in exactly this format:\n"
    '{"label": "<short noun phrase for that object>", "bbox_2d": [x1, y1, x2, y2]}\n'
    "Rules:\n"
    "- bbox_2d is the bounding box [left, top, right, bottom] of the marked object, with "
    "coordinates NORMALIZED to the range 0-1000, where (0,0) is the top-left corner of "
    "the image and (1000,1000) is the bottom-right corner.\n"
    "- Make the box as TIGHT as possible around ONLY that one object. Do NOT include "
    "adjacent or nearby objects, and do NOT pad the box with extra background.\n"
    "- The bounding box must contain the red dot.\n"
    "- Judge purely from where the red dot appears in the image you see.\n"
    "- Output the JSON only. No explanation, no markdown code fences."
)


# --- result type ---------------------------------------------------------------


@dataclass
class GroundedObject:
    """A grounded object with its bbox already converted to ORIGINAL image pixels."""

    label: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in original pixels
    source: str = ""                 # "text" | "point"
    input: str = ""                  # the user prompt or "[x, y]"
    raw: str = ""                    # raw VLM text, for debugging
    coord_hypothesis: str = ""       # which coordinate hypothesis was used
    bbox_model: tuple[int, int, int, int] | None = None  # raw bbox as emitted

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox": list(self.bbox),
            "source": self.source,
            "input": self.input,
            "raw": self.raw,
            "coord_hypothesis": self.coord_hypothesis,
            "bbox_model": list(self.bbox_model) if self.bbox_model else None,
        }


class GroundingParseError(ValueError):
    """Raised when the VLM output cannot be parsed into a label + bbox."""

    def __init__(self, raw: str):
        super().__init__(f"could not parse grounding from VLM output: {raw!r}")
        self.raw = raw


# --- parsing -------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_OBJ_RE_ALL = re.compile(r"\{[^{}]*bbox_2d[^{}]*\}", re.S)
_FOUR_INTS_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]*)"')

# Words that begin a relative/relational clause; we cut the label at the first one so
# only the head noun phrase (the object's generic name) survives.
_CLAUSE_MARKERS = re.compile(
    r"\b(that|which|who|whom|whose|wearing|worn|ridden|riding|driven|driving|holding|"
    r"held|carrying|sitting|seated|standing|lying|located|positioned|placed|labeled|"
    r"marked|next|behind|under|above|below|near|beside|with|in|on|at|to)\b",
    re.I,
)


def _object_phrase(label: str, max_words: int = 4) -> str:
    """Reduce a grounding label to a short object phrase suitable for FlowDIS.

    The VLM sometimes echoes the user's full disambiguating sentence into `label`
    (e.g. "the pedal kart ridden by the kid in a cyan shirt"). FlowDIS wants a clean
    object name on the already-cropped region, so we strip a leading article, cut at the
    first relational/relative-clause marker, and cap the word count. This is a safety
    net; the grounding prompt already asks for a short label, and an empty label is left
    empty (FlowDIS then runs unguided).
    """
    s = " ".join(label.split()).strip().strip('"').strip()
    if not s:
        return ""
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)
    m = _CLAUSE_MARKERS.search(s)
    if m and m.start() > 0:
        s = s[: m.start()].strip().rstrip(",")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s or " ".join(label.split()).strip()


def parse_grounding(raw: str) -> dict:
    """Parse a VLM grounding response into {"label": str, "bbox_2d": [4 floats]}.

    Tolerant of code fences and minor rambling. Raises GroundingParseError on failure
    rather than fabricating a box.
    """
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # 1) load the LAST {...} block that mentions bbox_2d (the final answer for a
    #    thinking model that may mention coordinates while reasoning first)
    for m in reversed(_OBJ_RE_ALL.findall(text)):
        try:
            obj = json.loads(m)
            bbox = obj.get("bbox_2d")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return {
                    "label": _object_phrase(str(obj.get("label", "")).strip()),
                    "bbox_2d": [float(v) for v in bbox],
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # 2) fallback: regex the four numbers + an optional label anywhere in the text
    nums = _FOUR_INTS_RE.search(text)
    if nums:
        label_m = _LABEL_RE.search(text)
        return {
            "label": _object_phrase(label_m.group(1).strip() if label_m else ""),
            "bbox_2d": [float(nums.group(i)) for i in range(1, 5)],
        }

    raise GroundingParseError(raw)


# --- coordinate conversion -----------------------------------------------------


def resized_hw_from_grid(grid_thw, patch_size: int) -> tuple[int, int]:
    """Recover the model's input pixel dims (H, W) from image_grid_thw.

    image_grid_thw = (t, h, w) in patch units; resized pixels = grid * patch_size.
    """
    _, gh, gw = (int(v) for v in grid_thw)
    return gh * patch_size, gw * patch_size


def _clamp_box(b, W, H):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        max(0, min(W, int(round(x1)))),
        max(0, min(H, int(round(y1)))),
        max(0, min(W, int(round(x2)))),
        max(0, min(H, int(round(y2)))),
    )


def _is_valid(b, W, H) -> bool:
    """In-bounds (with slack) and sane area (0.1%..95% of the image)."""
    bx1, by1, bx2, by2 = b
    if bx2 <= bx1 or by2 <= by1:
        return False
    if not (bx1 >= -0.05 * W and by1 >= -0.05 * H
            and bx2 <= 1.05 * W and by2 <= 1.05 * H):
        return False
    cb = _clamp_box(b, W, H)
    frac = (cb[2] - cb[0]) * (cb[3] - cb[1]) / float(W * H)
    return 0.001 <= frac <= 0.95


def norm1000_bbox_to_orig(
    bbox_model,
    orig_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], str]:
    """Convert a bbox given in normalized 0-1000 coords to original-image pixels.

    Used for cloud VLMs (OpenAI-compatible or Gemini API) which we explicitly ask for 0-1000 normalized
    coordinates, so no resized-space / image_grid_thw machinery is needed. Auto-falls
    back to treating coords as already-absolute pixels if the 0-1000 reading is
    degenerate (e.g. a model that ignored the instruction and returned pixels).
    """
    W, H = orig_size
    x1, y1, x2, y2 = bbox_model
    norm = (x1 * W / 1000.0, y1 * H / 1000.0, x2 * W / 1000.0, y2 * H / 1000.0)
    if _is_valid(norm, W, H):
        return _clamp_box(norm, W, H), "norm_1000"
    if _is_valid((x1, y1, x2, y2), W, H):
        return _clamp_box((x1, y1, x2, y2), W, H), "orig_abs"
    return _clamp_box(norm, W, H), "norm_1000_fallback"


def model_bbox_to_orig(
    bbox_model,
    grid_thw,
    patch_size: int,
    orig_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], str]:
    """Convert a VLM-emitted bbox to original-image pixels.

    Empirically, the local qwen3.5-27b checkpoint emits NORMALIZED 0-1000 coordinates
    (confirmed by calibration: a left-tower box of x 497-603 maps to 0.50-0.60 of width).
    So "norm_1000" is the primary hypothesis. Fallbacks handle other checkpoints:
      - "resized_abs": absolute pixels in the smart-resized input space (Qwen2.5-VL);
        only chosen when coords exceed 1000 so norm_1000 yields an out-of-bounds box.
      - "orig_abs":    coordinates already in original pixels.
    Tried in priority order; the first valid hypothesis wins. Returns
    (bbox_in_orig_pixels, hypothesis_name).
    """
    W, H = orig_size
    rh, rw = resized_hw_from_grid(grid_thw, patch_size)
    x1, y1, x2, y2 = bbox_model

    candidates = [
        ("norm_1000", (x1 * W / 1000.0, y1 * H / 1000.0, x2 * W / 1000.0, y2 * H / 1000.0)),
        ("resized_abs", (x1 * W / rw, y1 * H / rh, x2 * W / rw, y2 * H / rh)),
        ("orig_abs", (x1, y1, x2, y2)),
    ]
    for name, b in candidates:
        if _is_valid(b, W, H):
            return _clamp_box(b, W, H), name
    # nothing clearly valid: fall back to the current trained convention, flagged for review
    return _clamp_box(candidates[0][1], W, H), "norm_1000_fallback"
