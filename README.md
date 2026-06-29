# Any Prompt DIS — Dichotomous Image Segmentation from Any Prompt

**Any Prompt DIS** adds a vision-language **grounding** layer on top of
[FlowDIS](https://flowdis.github.io/), so you can isolate an object from a **complex
natural-language description**, a **single click**, or a **manual box** — not just a single
keyword. A VLM reasons about which object you mean and locates it (bounding box + a concise
*object prompt*); we crop that region and let FlowDIS produce a clean, high-detail matte.

> Built on top of FlowDIS (Picsart AI Research, CVPR 2026). The FlowDIS core under
> [`flowdis/`](flowdis/) is upstream; this project adds the [`agent/`](agent/) grounding
> layer and CLIs. See [Credits](#credits) and [License](#license).

<p align="center">
  <a href='https://arxiv.org/abs/2605.05077'><img src='https://img.shields.io/badge/FlowDIS-arXiv-red?logo=arXiv&logoColor=red'></a>&ensp;
  <a href='https://flowdis.github.io/'><img src='https://img.shields.io/badge/FlowDIS-Project%20Page-blue'></a>&ensp;
  <a href="https://huggingface.co/PAIR/FlowDIS"><img src="https://img.shields.io/badge/Weights-PAIR%2FFlowDIS-yellow?logo=huggingface"></a>
</p>

---

## Demos

> Full-quality MP4s are committed under [`assets/demo/`](assets/demo/) and embedded with
> `<video>` tags pointing at the raw files in this repo. (If a player doesn't render on your
> fork, drag-drop each `.mp4` into the README editor on github.com and paste the generated
> `user-attachments` URL — that also auto-embeds and is fully self-contained.)

**Text prompt** — disambiguate with language ("the tall bridge tower on the left, not the one on the right"):

https://github.com/user-attachments/assets/7e559a6a-c77f-4963-901b-cae39e994a95

**Point click** — click the object you want; the VLM grounds it, FlowDIS segments it:

https://github.com/user-attachments/assets/893d29e0-e0f9-42fc-be00-64464cdf985a

**Bounding box** — draw a box (optionally let the VLM auto-label it):

https://github.com/user-attachments/assets/941dcb42-2f87-491a-a724-2b9871553425

---

## How it works

```
  image + (text | click | box)
            │
            ▼
   ┌─────────────────────┐   bbox + object prompt
   │  VLM grounding       │ ───────────────────────┐
   │  (cloud or local)    │                         │
   └─────────────────────┘                         ▼
            │                              crop region (+ padding)
            │                                       │
            │                                       ▼
            │                              ┌──────────────────┐
            │                              │ FlowDIS segment  │
            │                              │ on the clean crop│
            │                              └──────────────────┘
            │                                       │
            ▼                                       ▼
   full-image mask  ◄───────────── paste crop mask back into place
```

Why crop first? FlowDIS takes a *single* phrase as its text condition (T5/CLIP) and has no
classifier-free guidance to amplify it, so it cannot resolve "the one on the *left*, not the
right." Letting a VLM ground the target and segmenting just that crop sidesteps the
ambiguity and gives FlowDIS a clean, tightly-framed input. See [`agent/pipeline.py`](agent/pipeline.py)
(`ground_and_segment`, `segment_grounded`).

---

## Installation

```bash
git clone https://github.com/Albertchen98/any-prompt-dis
cd any-prompt-dis
pip install -e .
```

Requirements: Python 3.10–3.12 and a CUDA GPU (FlowDIS needs **≥ 48 GB** for 1024², more at
higher resolution). FlowDIS weights download automatically from the Hugging Face Hub
([`PAIR/FlowDIS`](https://huggingface.co/PAIR/FlowDIS)) on first run.

## Grounding backend

The default backend is a **cloud VLM over an HTTP API** — it uses no local GPU/VRAM, so
FlowDIS stays resident and everything runs in one process. It speaks two request formats, so
you can point it at any **OpenAI-compatible** endpoint (OpenAI, OpenRouter, vLLM, Together, …)
or the **Google Gemini** native API. Configure with env vars (only the key is required):

```bash
export VLM_API_KEY=...                 # your API key (or write it to ~/.config/anyprompt-dis/api_key)
export VLM_API_FORMAT=openai           # "openai" (default) or "gemini"
export VLM_MODEL=google/gemini-3.1-pro-preview   # model id for your provider
# export VLM_API_BASE=https://api.openai.com/v1  # optional; defaults per format
# export VLM_PROXY=http://host:port              # optional, only if you need a proxy
```

For the Gemini native API:

```bash
export VLM_API_KEY=...
export VLM_API_FORMAT=gemini
export VLM_MODEL=gemini-2.0-flash      # base URL defaults to the Gemini endpoint
```

An **offline local VLM** backend ([`agent/vlm.py`](agent/vlm.py), Qwen-VL) is also available;
see [Local (offline) backend](#local-offline-vlm-backend-advanced).

## Quick start

### Interactive app (Gradio)

```bash
# FLOWDIS_DIR is optional; omit to auto-download weights from the HF Hub.
export VLM_API_KEY=...
python agent/gradio_app.py
```

Pick **Text prompt**, **Point click**, or **Bounding box**, and the app grounds → crops →
segments, returning a mask overlay, an RGBA cutout, and a grounding debug view.

### CLI — single image, text-grounded

```bash
python inference_grounded.py \
    --image-path assets/examples/1.jpg \
    --prompt "the tall bridge tower on the left, not the one on the right" \
    --output-path out/mask.png \
    --composite-path out/cutout.png \
    --debug-path out/debug.png
```

It prints `object_prompt=... bbox=[...]` and writes the mask / cutout / grounding overlay.
Use `--root-model-dir` for local FlowDIS weights, `--model` / `--api-format` to pick the
cloud model and request format.

### Library

```python
from PIL import Image
from flowdis.util import load_models
from agent.cloud_vlm import CloudVLM
from agent.pipeline import ground_and_segment

models = load_models(device="cuda")          # FlowDIS (weights auto-download)
vlm = CloudVLM()                               # cloud grounding (needs VLM_API_KEY)

image = Image.open("assets/examples/1.jpg").convert("RGB")
mask, grounded, bbox_padded = ground_and_segment(
    image, "the gold scissors, not the thread or tape", vlm, models,
)
print(grounded.label, grounded.bbox)           # object prompt + bounding box (orig pixels)
mask.save("mask.png")
```

---

## Plain FlowDIS (no grounding)

The original FlowDIS entry points still work for keyword/empty-prompt segmentation.

Batch over a directory (multi-GPU aware):

```bash
python inference.py \
    --images-dir /path/to/images \
    --output-dir /path/to/output \
    --prompts-json /path/to/prompts.json \
    --num-steps 2 --resolution 1024
```

Single image:

```bash
python inference_si.py --image-path input.jpg --prompt "" --output-path mask.png
```

Programmatic:

```python
from PIL import Image
from flowdis import flowdis_predict, load_models

models = load_models(device="cuda")
mask = flowdis_predict(
    image=Image.open("input.jpg").convert("RGB"),
    prompt="",                  # empty = unguided foreground segmentation
    models=models, resolution=1024, num_inference_steps=2, device="cuda",
)
mask.save("mask.png")
```

`--prompts-json` maps `{ "image.jpg": "a red sports car" }`. Pre-generated DIS prompts and
precomputed paper results: see the [FlowDIS repo](https://github.com/Picsart-AI-Research/FlowDIS).

## Local (offline) VLM backend (advanced)

To ground without a cloud call, use the local Qwen-VL backend. It needs the weights and a
recent transformers:

```bash
export QWEN_VLM_PATH=/path/to/qwen-vl-weights     # required for the local backend
pip install -U "transformers>=5"                  # local Qwen-VL needs transformers 5.x
```

The 27B VLM (~54 GB) and FlowDIS cannot co-reside on one 96 GB GPU, so the batch CLI runs
grounding in a child process first (freeing its VRAM) before loading FlowDIS:

```bash
python run_agent_seg.py --spec spec.json --output-dir out/ --stage all
# spec.json: {"1.jpg": {"text": "..."}, "4.jpg": {"point": [950, 700]}}
```

---

## Roadmap

- [x] Any prompt (bounding box, point, text prompt)
- [ ] Quantization of FlowDIS (runnable < 24 GB VRAM)
- [ ] Part segmentation
---

## Credits

This project is a grounding layer built on **FlowDIS** by Andranik Sargsyan and Shant
Navasardyan (Picsart AI Research, CVPR 2026). The FlowDIS core in [`flowdis/`](flowdis/) is
their work; FlowDIS itself builds on [FLUX.1 [schnell]](https://github.com/black-forest-labs/flux)
and [DIS5K](https://github.com/xuebinqin/DIS). If you use this work, please cite FlowDIS:

```bibtex
@article{sargsyan2026flowdis,
  title={{FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching}},
  author={Sargsyan, Andranik and Navasardyan, Shant},
  journal={arXiv preprint arXiv:2605.05077},
  year={2026},
  url={https://arxiv.org/abs/2605.05077}
}
```

## License

The FlowDIS core and weights are governed by the
[PicsArt Inc. FlowDIS Model License](LICENSE) — review it before any redistribution or
commercial use. The `agent/` additions in this repository are provided under the same terms.
