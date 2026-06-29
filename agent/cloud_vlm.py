"""Cloud VLM grounding over an HTTP API.

Provider-neutral client with the same interface as the local `agent.vlm.VLM`
(`ground_from_text` / `ground_from_point` -> GroundedObject). The model runs in the cloud,
so no GPU/RAM is used for grounding and FlowDIS can stay resident. We ask the model for
NORMALIZED 0-1000 coordinates, which are image-size-agnostic, so no local image-processor /
smart_resize machinery is needed.

Two request formats are supported, selected by `api_format` (or the VLM_API_FORMAT env var):

- ``"openai"`` — OpenAI-compatible ``/chat/completions`` (OpenAI, OpenRouter, vLLM,
  Together, LM Studio, …). Auth via ``Authorization: Bearer <key>``.
- ``"gemini"`` — Google Gemini native ``:generateContent``. Auth via ``x-goog-api-key``.

Configuration (env vars, all optional except the key):
    VLM_API_KEY       the API key (or write it to ~/.config/anyprompt-dis/api_key)
    VLM_API_FORMAT    "openai" (default) or "gemini"
    VLM_API_BASE      base URL; defaults per format (see _DEFAULT_BASES)
    VLM_MODEL         default model id
    VLM_PROXY         optional HTTP(S) proxy (default: direct)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path

import requests
from PIL import Image

from agent.grounding import (
    POINT_GROUNDING_PROMPT_NORM,
    TEXT_GROUNDING_PROMPT_NORM,
    GroundedObject,
    norm1000_bbox_to_orig,
    parse_grounding,
)
from agent.viz import draw_marker

logger = logging.getLogger(__name__)

# Request format and the default base URL for each.
DEFAULT_API_FORMAT = os.environ.get("VLM_API_FORMAT", "openai")
_DEFAULT_BASES = {
    "openai": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}
# Explicit override; if unset the base is chosen from the format above.
DEFAULT_API_BASE = os.environ.get("VLM_API_BASE") or None
DEFAULT_MODEL = os.environ.get("VLM_MODEL", "google/gemini-3.1-pro-preview")
# Optional HTTP(S) proxy. Defaults to direct (None). Set VLM_PROXY (or pass proxy=...) if you
# must route through a proxy. A TLS-intercepting proxy needs its CA in the system trust store.
DEFAULT_PROXY = os.environ.get("VLM_PROXY") or None
# Keep the image data small so the whole JSON request stays light (and proxy-friendly).
# Normalized 0-1000 coords map back to the full-res original even when the upload is small.
DEFAULT_MAX_SIDE = 768
DEFAULT_MAX_IMAGE_BYTES = 70_000
# Verify against the system CA bundle (covers a TLS-intercepting proxy whose CA is
# installed there); falls back to certifi if neither path exists.
_CA_CANDIDATES = [
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
]
KEY_FILE = Path("~/.config/anyprompt-dis/api_key").expanduser()
# Suggested models that do bbox grounding well; surfaced in the UI dropdown. Adjust to match
# your provider (these are OpenRouter-style ids; for Gemini-native use e.g. "gemini-2.0-flash").
GROUNDING_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-32b-instruct",
]

CROP_LABEL_PROMPT = (
    "You are labeling a manually selected image crop for an object segmentation model. "
    "Identify the main object inside this crop and output ONLY a JSON object in exactly "
    'this format: {"label": "<short noun phrase>"}\n'
    "Rules:\n"
    "- Use 1 to 5 words.\n"
    "- Name the object or object part, not the background.\n"
    "- Do not mention bounding boxes, coordinates, or uncertainty.\n"
    "- Output JSON only."
)


def _load_api_key() -> str:
    key = os.environ.get("VLM_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(
            "VLM API key not found. Set the VLM_API_KEY environment variable or write it "
            f"to {KEY_FILE}."
        )
    return key


def _ca_bundle() -> str | bool:
    for c in _CA_CANDIDATES:
        if os.path.exists(c):
            return c
    return True  # fall back to certifi


def _encode_image_b64(
    image: Image.Image,
    max_side: int = DEFAULT_MAX_SIDE,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> tuple[str, str]:
    """Downscale (keeping aspect) and return (base64_jpeg, mime_type).

    Downscaling is safe: we request normalized 0-1000 coords, so the box maps back to
    the ORIGINAL image regardless of what we upload. `max_bytes` budgets the base64 size.
    """
    src = image.convert("RGB")
    sides = [max_side, 640, 512, 448, 384, 320, 256]
    sides = [s for i, s in enumerate(sides) if s <= max_side and s not in sides[:i]]
    last_b64 = ""
    for side in sides:
        img = src.copy()
        img.thumbnail((side, side))
        for quality in (88, 80, 72, 64, 56):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            last_b64 = b64
            if len(b64) <= max_bytes:
                logger.debug("encoded image side<=%d quality=%d b64_bytes=%d", side, quality, len(b64))
                return b64, "image/jpeg"
    logger.warning("encoded image still exceeds budget: b64_bytes=%d budget=%d", len(last_b64), max_bytes)
    return last_b64, "image/jpeg"


class CloudVLM:
    """Cloud-backed grounding VLM with the agent.vlm.VLM interface.

    Talks to either an OpenAI-compatible endpoint (`api_format="openai"`) or the Google
    Gemini native API (`api_format="gemini"`).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        proxy: str | None = DEFAULT_PROXY,
        timeout: int = 90,
        max_side: int = DEFAULT_MAX_SIDE,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        api_format: str = DEFAULT_API_FORMAT,
        api_base: str | None = DEFAULT_API_BASE,
    ):
        self.model = model
        self.timeout = timeout
        self.max_side = max_side
        self.max_image_bytes = max_image_bytes
        self.api_format = api_format.lower()
        if self.api_format not in _DEFAULT_BASES:
            raise ValueError(
                f"unknown api_format {api_format!r}; expected one of {sorted(_DEFAULT_BASES)}"
            )
        self.api_base = (api_base or _DEFAULT_BASES[self.api_format]).rstrip("/")
        self._key = _load_api_key()
        self._ca = _ca_bundle()
        self._session = requests.Session()
        self._session.trust_env = False  # ignore inherited proxy env vars; use `proxy` only
        self._proxies = {"http": proxy, "https": proxy} if proxy else None
        logger.info(
            "CloudVLM ready (format=%s, base=%s, model=%s, proxy=%s, max_side=%d, max_image_bytes=%d)",
            self.api_format, self.api_base, model, proxy, max_side, max_image_bytes,
        )

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        resp = self._session.post(
            url, headers={"Content-Type": "application/json", **headers},
            json=body, proxies=self._proxies, verify=self._ca, timeout=self.timeout,
        )
        if resp.headers.get("content-type", "").startswith("text/html"):
            raise RuntimeError(
                f"VLM request returned an HTML error page (status {resp.status_code}). This "
                "usually means a proxy or the API rejected the request. Try lowering "
                "CloudVLM(max_side=...) or max_image_bytes=...."
            )
        resp.raise_for_status()
        return resp.json()

    def _chat(self, image: Image.Image, prompt: str, model: str | None = None) -> str:
        model = model or self.model
        b64, mime = _encode_image_b64(image, self.max_side, self.max_image_bytes)
        if self.api_format == "gemini":
            return self._chat_gemini(model, prompt, b64, mime)
        return self._chat_openai(model, prompt, b64, mime)

    def _chat_openai(self, model: str, prompt: str, b64: str, mime: str) -> str:
        body = {
            "model": model,
            "max_tokens": 1500,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
        }
        data = self._post(
            f"{self.api_base}/chat/completions",
            {"Authorization": f"Bearer {self._key}"},
            body,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected VLM (openai) response: {data}") from e

    def _chat_gemini(self, model: str, prompt: str, b64: str, mime: str) -> str:
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime, "data": b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1500},
        }
        data = self._post(
            f"{self.api_base}/models/{model}:generateContent",
            {"x-goog-api-key": self._key},
            body,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            if not text:
                raise KeyError("no text parts")
            return text
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected VLM (gemini) response: {data}") from e

    def ground_from_text(
        self, image: Image.Image, user_prompt: str, model: str | None = None
    ) -> GroundedObject:
        raw = self._chat(image, TEXT_GROUNDING_PROMPT_NORM.format(user_prompt=user_prompt), model)
        parsed = parse_grounding(raw)
        bbox, hyp = norm1000_bbox_to_orig(parsed["bbox_2d"], image.size)
        logger.info("ground_from_text(%s) label=%r bbox=%s", model or self.model, parsed["label"], bbox)
        return GroundedObject(
            label=parsed["label"], bbox=bbox, source="text", input=user_prompt,
            raw=raw, coord_hypothesis=hyp,
            bbox_model=tuple(int(round(v)) for v in parsed["bbox_2d"]),
        )

    def ground_from_point(
        self, image: Image.Image, point: tuple[int, int], model: str | None = None
    ) -> GroundedObject:
        marked = draw_marker(image, point)
        raw = self._chat(marked, POINT_GROUNDING_PROMPT_NORM, model)
        parsed = parse_grounding(raw)
        bbox, hyp = norm1000_bbox_to_orig(parsed["bbox_2d"], image.size)
        x, y = int(point[0]), int(point[1])
        contains = bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
        if not contains:
            logger.warning("grounded bbox %s does NOT contain click %s", bbox, point)
        logger.info("ground_from_point(%s) label=%r bbox=%s contains=%s",
                    model or self.model, parsed["label"], bbox, contains)
        return GroundedObject(
            label=parsed["label"], bbox=bbox, source="point", input=f"[{x}, {y}]",
            raw=raw, coord_hypothesis=hyp,
            bbox_model=tuple(int(round(v)) for v in parsed["bbox_2d"]),
        )

    def label_crop(self, image: Image.Image, model: str | None = None) -> tuple[str, str]:
        """Return a short label for a manually selected crop."""
        raw = self._chat(image, CROP_LABEL_PROMPT, model)
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        try:
            obj = json.loads(text)
            label = str(obj.get("label", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
            label = m.group(1).strip() if m else text.strip().strip('"')
        label = " ".join(label.split())
        if not label:
            raise RuntimeError(f"VLM did not return a usable crop label. Raw: {raw[:200]}")
        logger.info("label_crop(%s) label=%r", model or self.model, label)
        return label, raw
