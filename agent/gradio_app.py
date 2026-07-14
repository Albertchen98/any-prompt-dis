"""Gradio demo: VLM-grounded crop-then-segment on top of FlowDIS.

Two modes:
  - Text:  type a complex/disambiguating prompt ("the cup on the table, not the one on
           the stove"); a cloud VLM (OpenAI-compatible or Gemini API) reasons + grounds it.
  - Point: click the object in the image; the VLM confirms which object is under the dot.
In both cases we crop the grounded region and run FlowDIS on the clean crop, then paste
the mask back into a full-image result.
A "Plain (no VLM)" mode runs FlowDIS directly on the full image with the raw prompt —
no cloud call, no crop — as a baseline / offline fallback.

Only FlowDIS runs locally (resident on GPU); grounding is a cloud API call, so there
is no GPU/RAM contention and the app stays responsive.

Run:  python agent/gradio_app.py     (from the repo root, or: PYTHONPATH=. python agent/gradio_app.py)
"""

import logging
import os
import re
import sys
import uuid
import hashlib
from pathlib import Path

# Make `import agent...` work whether launched as `agent/gradio_app.py` or `-m`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("flowdis_agent_demo")

TEMP_DIR = _REPO_ROOT / "agent" / "gradio_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR))

import gradio as gr
from PIL import Image

from agent.cloud_vlm import DEFAULT_MODEL, GROUNDING_MODELS, CloudVLM
from agent.crop import pad_and_clamp
from agent.grounding import GroundedObject, GroundingParseError
from agent.pipeline import segment_grounded
from agent.viz import draw_debug, draw_marker, to_mask_overlay, to_transparent_png
from flowdis.sampling import flowdis_predict

FLOWDIS_DIR = os.environ.get("FLOWDIS_DIR")  # None -> auto-download from HF Hub (PAIR/FlowDIS)
DEVICE = os.environ.get("FLOWDIS_DEVICE", "cuda")
INT8 = os.environ.get("FLOWDIS_INT8", "0").lower() in ("1", "true", "yes")
T5_INT4 = os.environ.get("FLOWDIS_T5_INT4", "0").lower() in ("1", "true", "yes")

# --- load models once ----------------------------------------------------------
logger.info("Loading FlowDIS (resident)...")
from flowdis.util import load_models

MODELS = load_models(
    root_model_dir=Path(FLOWDIS_DIR) if FLOWDIS_DIR else None,
    device=DEVICE,
    int8=INT8,
    t5_int4=T5_INT4,
)
# Grounding backend: "cloud" (default, API key needed) or "local" (Qwen-VL weights on GPU,
# set QWEN_VLM_PATH; shares the GPU with FlowDIS, so mind VRAM).
VLM_BACKEND = os.environ.get("VLM_BACKEND", "cloud").lower()
if VLM_BACKEND == "local":
    from agent.vlm import VLM as LocalVLM

    logger.info("Loading local VLM (resident)...")
    VLM = LocalVLM(device=DEVICE)
else:
    # The cloud VLM needs an API key; without one the app still serves "Plain (no VLM)" mode.
    try:
        VLM = CloudVLM(model=DEFAULT_MODEL)
    except RuntimeError as e:
        logger.warning("Cloud VLM unavailable (%s) — only Plain (no VLM) mode will work.", e)
        VLM = None
logger.info("Ready.")

# Local Qwen-VL boxes are noticeably less tight than Gemini's, so pad the crop more
# by default to make sure the whole object survives into the FlowDIS crop.
PAD_FRAC_DEFAULT = 0.22 if VLM_BACKEND == "local" else 0.12


def _image_fingerprint(image):
    """Cheap identity for detecting stale Gradio state after examples/image swaps."""
    if image is None:
        return None
    img = image.convert("RGB")
    h = hashlib.blake2b(digest_size=12)
    h.update(str(img.size).encode("ascii"))
    h.update(img.resize((32, 32)).tobytes())
    return h.hexdigest()


def _point_xy(point):
    if isinstance(point, dict):
        return point.get("xy")
    return point


def on_upload(image):
    """New image: remember the clean original and clear any previous click."""
    return image, _image_fingerprint(image), None


def on_example(image, mode, prompt):
    """Examples update component values, so sync hidden clean-image state too."""
    if image is None:
        return None, None, None, ""
    return image, _image_fingerprint(image), None, "Loaded example."


def on_select(input_img, orig_img, orig_fp, point, mode, evt: gr.SelectData):
    """Record a click as the target point and DRAW it on the image so the user sees it.

    We keep the clean original in `orig_state` (segmentation must not see the red dot) and
    show a marked copy in the image box for visual feedback.
    """
    if mode != "Point click":
        if mode == "Bounding box":
            status = "Bounding box mode uses the box input only. Switch to Point click to select by clicking."
        elif mode == "Plain (no VLM)":
            status = "Plain mode runs FlowDIS on the full image with the prompt only. Switch to Point click to select by clicking."
        else:
            status = "Text mode uses the prompt only. Switch to Point click to select by clicking."
        return (
            gr.update(),
            point,
            orig_img,
            orig_fp,
            status,
        )

    input_fp = _image_fingerprint(input_img)
    old_marked_fp = point.get("marked_fp") if isinstance(point, dict) else None
    old_orig_fp = point.get("orig_fp") if isinstance(point, dict) else None
    base = orig_img if (
        orig_img is not None
        and (input_fp == orig_fp or input_fp == old_marked_fp or orig_fp == old_orig_fp)
    ) else input_img
    if base is None:
        return gr.update(), None, None, None, "Upload an image first."
    point = [int(evt.index[0]), int(evt.index[1])]
    base_fp = _image_fingerprint(base)
    marked = draw_marker(base.convert("RGB"), tuple(point))
    point_meta = {"xy": point, "orig_fp": base_fp, "marked_fp": _image_fingerprint(marked)}
    status = f"**Clicked point:** ({point[0]}, {point[1]}) — shown as the red dot. Press *Segment*."
    return marked, point_meta, base, base_fp, status


def _clean_image_for_run(image, orig_img, orig_fp, point):
    """Prefer the clean original for point runs, even if Gradio re-encoded the marked view."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    display_fp = _image_fingerprint(image)
    marked_fp = point.get("marked_fp") if isinstance(point, dict) else None
    point_orig_fp = point.get("orig_fp") if isinstance(point, dict) else None
    if orig_img is not None and orig_fp is not None and orig_fp == point_orig_fp:
        return orig_img.convert("RGB"), orig_fp
    if orig_img is not None and (display_fp == orig_fp or display_fp == marked_fp):
        return orig_img.convert("RGB"), orig_fp
    return image.convert("RGB"), display_fp


def _bbox_contains_point(bbox, point):
    x, y = int(point[0]), int(point[1])
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _parse_manual_bbox(text, image_size):
    """Parse a manual bbox in original-image pixels."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if len(nums) != 4:
        raise gr.Error("Bounding box mode: enter exactly four numbers: x1, y1, x2, y2.")
    x1, y1, x2, y2 = (int(round(float(v))) for v in nums)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    W, H = image_size
    bbox = (
        max(0, min(W, x1)),
        max(0, min(H, y1)),
        max(0, min(W, x2)),
        max(0, min(H, y2)),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise gr.Error(f"Bounding box mode: degenerate box after clamping: {bbox}.")
    return bbox


def _require_vlm():
    if VLM is None:
        raise gr.Error(
            "Cloud VLM is not configured (no API key found). Set VLM_API_KEY, or use "
            "Plain (no VLM) mode / Bounding box mode with a manual prompt."
        )
    return VLM


def _ground_from_text(image, text, model):
    v = _require_vlm()
    # The local backend is a single fixed checkpoint; only the cloud one takes a model id.
    if VLM_BACKEND == "local":
        return v.ground_from_text(image, text)
    return v.ground_from_text(image, text, model=model)


def _ground_from_point(image, point_xy, model):
    v = _require_vlm()
    if VLM_BACKEND == "local":
        return v.ground_from_point(image, point_xy)
    return v.ground_from_point(image, point_xy, model=model)


def _label_crop(crop, model):
    v = _require_vlm()
    if VLM_BACKEND == "local":
        raise gr.Error(
            "The local VLM backend has no crop auto-label; type a FlowDIS prompt for the box."
        )
    return v.label_crop(crop, model=model)


def _run(image, orig_img, orig_fp, mode, prompt, bbox_text, point, model, resolution, num_steps, pad_frac):
    # Generator: yields intermediate UI updates so the grounding result shows up
    # as soon as the VLM answers, instead of loading together with FlowDIS.
    # Use the clean original when the displayed image is the red-dot marked copy.
    image, clean_fp = _clean_image_for_run(image, orig_img, orig_fp, point)

    # Plain mode: FlowDIS on the full image with the raw prompt — no VLM, no crop.
    if mode == "Plain (no VLM)":
        yield gr.update(), None, "⏳ **FlowDIS** running on the full image (no VLM)…", gr.update()
        full_mask = flowdis_predict(
            image=image, prompt=(prompt or "").strip(), models=MODELS,
            resolution=int(resolution), num_inference_steps=int(num_steps), device=DEVICE,
        )
        preview = to_mask_overlay(image, full_mask)
        composite = to_transparent_png(image, full_mask)
        png_path = TEMP_DIR / f"{uuid.uuid4().hex}.png"
        composite.save(png_path)
        status = (
            f"**Mode:** plain FlowDIS (no VLM, no crop)  \n"
            f"**prompt:** {prompt.strip() if (prompt or '').strip() else '(empty — salient foreground)'}  \n"
            f"**sizes:** image={image.size}, mask={full_mask.size}, preview={preview.size}"
        )
        yield (image, preview), None, status, gr.update(value=str(png_path), interactive=True)
        return

    # 1) ground the target with the VLM
    yield gr.update(), gr.update(), f"⏳ **Stage 1/2 — VLM grounding** ({'local' if VLM_BACKEND == 'local' else model})…", gr.update()
    try:
        if mode == "Bounding box":
            bbox = _parse_manual_bbox(bbox_text, image.size)
            manual_label = (prompt or "").strip()
            if manual_label:
                label = manual_label
                raw = ""
                coord_hypothesis = "manual original pixels; manual prompt"
            else:
                label, raw = _label_crop(image.crop(bbox), model)
                coord_hypothesis = "manual original pixels; VLM crop label"
            g = GroundedObject(
                label=label,
                bbox=bbox,
                source="bbox",
                input=bbox_text,
                raw=raw,
                coord_hypothesis=coord_hypothesis,
                bbox_model=bbox,
            )
        elif mode == "Point click":
            point_xy = _point_xy(point)
            if not point_xy:
                raise gr.Error("Point mode: click the object in the image first.")
            if isinstance(point, dict) and point.get("orig_fp") != clean_fp:
                raise gr.Error("Point mode: the image changed after the click. Click the target again.")
            g = _ground_from_point(image, tuple(point_xy), model)
            if not _bbox_contains_point(g.bbox, point_xy):
                raise gr.Error(
                    f"Grounding box {g.bbox} does not contain the clicked point {tuple(point_xy)}. "
                    "Try clicking closer to the object center or use text mode."
                )
        else:
            if not (prompt and prompt.strip()):
                raise gr.Error("Text mode: enter a prompt describing the target.")
            g = _ground_from_text(image, prompt.strip(), model)
    except GroundingParseError as e:
        raise gr.Error(f"VLM did not return a usable box. Raw: {e.raw[:200]}")
    except gr.Error:
        raise
    except Exception as e:  # network/proxy/etc
        raise gr.Error(f"Grounding failed: {e}")

    # Grounding done — show the debug box right away, before FlowDIS starts.
    bbox_pad = pad_and_clamp(g.bbox, image.size, pad_frac=pad_frac)
    debug = draw_debug(
        image, bbox_raw=g.bbox, bbox_padded=bbox_pad,
        point=tuple(_point_xy(point)) if (mode == "Point click" and _point_xy(point)) else None,
        label=g.label,
    )
    yield (
        gr.update(), debug,
        f"✅ **Stage 1/2 — grounded:** {g.label or '(unnamed)'} · bbox {g.bbox}  \n"
        f"⏳ **Stage 2/2 — FlowDIS** segmenting the crop…",
        gr.update(),
    )

    # 2) crop (with padding) -> FlowDIS -> paste back. Shared with the CLI / library
    # so the object-prompt + crop contract lives in one place (agent.pipeline).
    full_mask, _ = segment_grounded(
        image, g, MODELS,
        resolution=int(resolution), num_steps=int(num_steps),
        pad_frac=pad_frac, device=DEVICE,
    )

    # 3) build outputs
    preview = to_mask_overlay(image, full_mask)     # same-size overlay for pixel-aligned viewing
    composite = to_transparent_png(image, full_mask)  # RGBA cutout, for download
    png_path = TEMP_DIR / f"{uuid.uuid4().hex}.png"
    composite.save(png_path)

    if mode == "Bounding box" and (prompt or "").strip():
        model_str = "manual bbox prompt (no VLM)"
    elif VLM_BACKEND == "local":
        model_str = f"local: {VLM.model_path}"
    else:
        model_str = model
    status = (
        f"**Target:** {g.label or '(unnamed)'}  \n"
        f"**bbox (orig px):** {g.bbox}  ·  **coord mode:** {g.coord_hypothesis}  \n"
        f"**sizes:** image={image.size}, mask={full_mask.size}, preview={preview.size}  \n"
        f"**model:** {model_str}"
    )
    yield (image, preview), debug, status, gr.update(value=str(png_path), interactive=True)


def _toggle_mode(mode, image, orig_img, orig_fp, point):
    is_point = mode == "Point click"
    is_box = mode == "Bounding box"
    is_plain = mode == "Plain (no VLM)"
    restored = gr.update()
    status = ""
    if not is_point:
        marked_fp = point.get("marked_fp") if isinstance(point, dict) else None
        if orig_img is not None and _image_fingerprint(image) == marked_fp:
            restored = orig_img
        point = None
        if is_box:
            status = "Bounding box mode active: drag on the image to fill the box, or type x1, y1, x2, y2 manually."
        elif is_plain:
            status = "Plain mode active: FlowDIS runs on the full image with the prompt below — no VLM call, no crop."
        else:
            status = "Text mode active: clicks are ignored; segmentation uses the prompt."
    if is_box:
        prompt_label = "FlowDIS prompt (optional)"
        prompt_placeholder = "optional; leave blank to ask Gemini to label the box"
    elif is_plain:
        prompt_label = "FlowDIS prompt (optional)"
        prompt_placeholder = "e.g. the gold scissors; leave blank for the salient foreground object"
    else:
        prompt_label = "Text prompt"
        prompt_placeholder = "e.g. the cup on the table, not the one on the stove"
    return (
        gr.update(
            visible=not is_point,
            label=prompt_label,
            placeholder=prompt_placeholder,
        ),                                # prompt box
        gr.update(visible=is_point),      # point hint
        gr.update(visible=is_box),        # manual bbox box
        gr.update(
            visible=not is_plain and VLM_BACKEND != "local",
            label="VLM for grounding/auto-label" if is_box else "Grounding VLM (cloud API)",
        ),                                # VLM model dropdown (local backend: fixed checkpoint)
        gr.update(visible=not is_plain),  # crop padding slider
        point,
        restored,
        status,
    )


_CSS = """
#agent-input img,
#agent-output img,
#agent-output canvas,
#agent-debug img {
  object-fit: contain !important;
}
#agent-output {
  max-height: 460px;
}
#agent-bbox-drag-layer {
  border: 1px dashed rgba(0, 180, 255, 0.9);
  box-sizing: border-box;
  cursor: crosshair;
  display: none;
  position: fixed;
  z-index: 40;
}
#agent-bbox-drag-layer .agent-bbox-rect {
  background: rgba(0, 180, 255, 0.16);
  border: 2px solid rgb(0, 180, 255);
  box-sizing: border-box;
  display: none;
  position: absolute;
}
"""


_HEAD_JS = """
<script>
(function() {
  var layer = null;
  var rect = null;
  var dragging = false;
  var start = null;

  function findInputImage() {
    return document.querySelector('#agent-input img');
  }

  function findBboxInput() {
    return document.querySelector('#agent-bbox textarea, #agent-bbox input');
  }

  function isBboxMode() {
    var checked = document.querySelector('#agent-mode input[type=radio]:checked');
    if (!checked) return false;
    var value = checked.value || '';
    var label = checked.closest('label');
    var text = label ? label.innerText : '';
    return value.indexOf('Bounding box') >= 0 || text.indexOf('Bounding box') >= 0;
  }

  function ensureLayer() {
    if (layer) return layer;
    layer = document.createElement('div');
    layer.id = 'agent-bbox-drag-layer';
    rect = document.createElement('div');
    rect.className = 'agent-bbox-rect';
    layer.appendChild(rect);
    document.body.appendChild(layer);
    layer.addEventListener('pointerdown', onPointerDown);
    layer.addEventListener('pointermove', onPointerMove);
    layer.addEventListener('pointerup', onPointerUp);
    layer.addEventListener('pointercancel', onPointerUp);
    return layer;
  }

  function imageDrawRect(img) {
    if (!img || !img.naturalWidth || !img.naturalHeight) return null;
    var r = img.getBoundingClientRect();
    var scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    var w = img.naturalWidth * scale;
    var h = img.naturalHeight * scale;
    return {
      left: r.left + (r.width - w) / 2,
      top: r.top + (r.height - h) / 2,
      width: w,
      height: h,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight
    };
  }

  function syncLayer() {
    ensureLayer();
    var img = findInputImage();
    var d = imageDrawRect(img);
    if (!d || !isBboxMode()) {
      layer.style.display = 'none';
      layer.style.pointerEvents = 'none';
      return;
    }
    layer.style.display = 'block';
    layer.style.pointerEvents = 'auto';
    layer.style.left = d.left + 'px';
    layer.style.top = d.top + 'px';
    layer.style.width = d.width + 'px';
    layer.style.height = d.height + 'px';
  }

  function toImagePoint(evt) {
    var img = findInputImage();
    var d = imageDrawRect(img);
    if (!d) return null;
    var x = Math.max(0, Math.min(d.width, evt.clientX - d.left));
    var y = Math.max(0, Math.min(d.height, evt.clientY - d.top));
    return {
      x: Math.round(x * d.naturalWidth / d.width),
      y: Math.round(y * d.naturalHeight / d.height)
    };
  }

  function setTextValue(input, value) {
    var proto = Object.getPrototypeOf(input);
    var desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) desc.set.call(input, value);
    else input.value = value;
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }

  function updateRect(p) {
    if (!start || !p || !rect) return;
    var img = findInputImage();
    var d = imageDrawRect(img);
    if (!d) return;
    var x1 = Math.min(start.x, p.x);
    var y1 = Math.min(start.y, p.y);
    var x2 = Math.max(start.x, p.x);
    var y2 = Math.max(start.y, p.y);
    var sx = d.width / d.naturalWidth;
    var sy = d.height / d.naturalHeight;
    rect.style.display = 'block';
    rect.style.left = (x1 * sx) + 'px';
    rect.style.top = (y1 * sy) + 'px';
    rect.style.width = Math.max(1, (x2 - x1) * sx) + 'px';
    rect.style.height = Math.max(1, (y2 - y1) * sy) + 'px';
    return [x1, y1, x2, y2];
  }

  function onPointerDown(evt) {
    if (!isBboxMode()) return;
    start = toImagePoint(evt);
    if (!start) return;
    dragging = true;
    layer.setPointerCapture(evt.pointerId);
    rect.style.display = 'none';
    evt.preventDefault();
    evt.stopPropagation();
  }

  function onPointerMove(evt) {
    if (!dragging) return;
    updateRect(toImagePoint(evt));
    evt.preventDefault();
    evt.stopPropagation();
  }

  function onPointerUp(evt) {
    if (!dragging) return;
    dragging = false;
    var box = updateRect(toImagePoint(evt));
    if (box && Math.abs(box[2] - box[0]) >= 2 && Math.abs(box[3] - box[1]) >= 2) {
      var input = findBboxInput();
      if (input) setTextValue(input, box.join(', '));
    }
    evt.preventDefault();
    evt.stopPropagation();
  }

  function init() {
    ensureLayer();
    syncLayer();
    window.addEventListener('resize', syncLayer);
    window.addEventListener('scroll', syncLayer, true);
    setInterval(syncLayer, 250);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
"""


with gr.Blocks(title="FlowDIS Agent – Grounded Segmentation", css=_CSS, head=_HEAD_JS) as demo:
    gr.Markdown(
        "## FlowDIS Agent — language- & click-grounded segmentation\n"
        "A cloud VLM grounds your target (disambiguating text **or** a click), then FlowDIS "
        "segments just that object. Only FlowDIS runs locally."
    )
    point_state = gr.State(None)
    orig_state = gr.State(None)  # clean original image (without the click marker)
    orig_fp_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input image", type="pil", height=460, elem_id="agent-input")
            mode = gr.Radio(
                ["Text prompt", "Point click", "Bounding box", "Plain (no VLM)"],
                value="Text prompt",
                label="Grounding mode",
                info="Text/Point: VLM grounds target. Bounding box: manual box, optional VLM auto-label. "
                     "Plain: FlowDIS on the full image, no VLM.",
                elem_id="agent-mode",
            )
            prompt_box = gr.Textbox(
                label="Text prompt",
                placeholder="e.g. the cup on the table, not the one on the stove",
                lines=2, visible=True,
            )
            point_hint = gr.Markdown("Click the target object in the image above.", visible=False)
            bbox_box = gr.Textbox(
                label="Bounding box (original image pixels)",
                placeholder="x1, y1, x2, y2",
                lines=1,
                visible=False,
                elem_id="agent-bbox",
            )
            model_dd = gr.Dropdown(
                GROUNDING_MODELS, value=DEFAULT_MODEL, label="Grounding / label VLM (cloud API)",
                visible=VLM_BACKEND != "local",
            )
            with gr.Row():
                resolution = gr.Slider(512, 2048, value=512, step=64, label="FlowDIS resolution",
                                       info="Detailed/background-heavy images need 1536-2048.")
                num_steps = gr.Slider(1, 12, value=2, step=1, label="Steps",
                                      info="More steps = sharper, more stable masks.")
            pad_frac = gr.Slider(0.0, 0.4, value=PAD_FRAC_DEFAULT, step=0.01, label="Crop padding",
                                 info="Context around the grounded box. Local VLM boxes are less "
                                      "accurate, so the local backend defaults to more padding.")
            run_btn = gr.Button("Segment", variant="primary")

        with gr.Column(scale=1):
            output_slider = gr.ImageSlider(
                label="Original ↔ Mask overlay",
                type="pil",
                max_height=460,
                elem_id="agent-output",
                slider_position=50,
            )
            status = gr.Markdown()
            debug_image = gr.Image(label="Grounding (green=box, cyan=padded crop, red=click)",
                                   type="pil", height=320, elem_id="agent-debug")
            download_btn = gr.DownloadButton("Download cutout PNG", interactive=False)

    gr.Examples(
        examples=[
            ["assets/examples/1.jpg", "Text prompt",
             "the tall bridge tower on the left, not the one further to the right"],
            ["assets/examples/4.jpg", "Text prompt", "the gold scissors, not the thread or tape"],
        ],
        inputs=[input_image, mode, prompt_box],
        outputs=[orig_state, orig_fp_state, point_state, status],
        fn=on_example,
        run_on_click=True,
        label="Text examples",
    )

    mode.change(
        _toggle_mode,
        inputs=[mode, input_image, orig_state, orig_fp_state, point_state],
        outputs=[prompt_box, point_hint, bbox_box, model_dd, pad_frac, point_state, input_image, status],
    )
    input_image.select(
        on_select, inputs=[input_image, orig_state, orig_fp_state, point_state, mode],
        outputs=[input_image, point_state, orig_state, orig_fp_state, status],
    )
    input_image.upload(on_upload, inputs=input_image, outputs=[orig_state, orig_fp_state, point_state])

    run_btn.click(
        lambda: gr.update(interactive=False), outputs=download_btn,
    ).then(
        _run,
        inputs=[input_image, orig_state, orig_fp_state, mode, prompt_box, bbox_box, point_state, model_dd,
                resolution, num_steps, pad_frac],
        outputs=[output_slider, debug_image, status, download_btn],
        concurrency_limit=1,
        concurrency_id="flowdis_gpu",
    )


if __name__ == "__main__":
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        allowed_paths=[str(TEMP_DIR), str(_REPO_ROOT / "assets")],
    )
