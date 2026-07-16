
import torch
from safetensors.torch import load_file

from flowdis.autoencoder import AutoEncoder
from flowdis.conditioner import HFEmbedder
from flowdis.configs import configs
from flowdis.model import Flux, FluxParams
from flowdis.quant import is_quantized_state_dict, swap_quantized_linears


def _remap_clip_state_dict(
    clip: HFEmbedder,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Adapt CLIP checkpoint keys to the installed Transformers layout.

    FlowDIS's checkpoint uses ``hf_module.text_model.*`` keys.  In
    Transformers 5, ``CLIPTextModel`` exposes the same modules directly under
    ``hf_module.*``.  Older Transformers releases still expect the checkpoint
    layout, so only remap a key when the original key is not expected and the
    stripped key is.
    """
    expected_keys = set(clip.state_dict())
    checkpoint_prefix = "hf_module.text_model."
    module_prefix = "hf_module."
    remapped = {}

    for key, value in state_dict.items():
        candidate = (
            module_prefix + key[len(checkpoint_prefix):]
            if key.startswith(checkpoint_prefix)
            else key
        )
        target_key = candidate if key not in expected_keys and candidate in expected_keys else key
        if target_key in remapped:
            raise ValueError(f"Duplicate CLIP key after remapping: {target_key}")
        remapped[target_key] = value

    return remapped


def load_transformer(
    model_name: str,
    model_path: str,
    device: str | torch.device = "cuda",
    config: FluxParams = None,
    state_dict: dict = None,
) -> Flux:
    with torch.device("meta"):
        model = Flux(config if config else configs[model_name]).to(dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    if state_dict is None:
        if str(model_path).endswith(".safetensors"):
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu")
    if is_quantized_state_dict(state_dict):
        # INT8 ConvRot checkpoint (ComfyUI-native format, made by ctq): swap the
        # quantized Linears, then move without a dtype cast so the int8 weights
        # and fp32 scales survive; the remaining params are already bf16.
        swap_quantized_linears(model, state_dict)
        model.load_state_dict(state_dict, assign=True, strict=False)
        model = model.to(device=device)
    else:
        model.load_state_dict(state_dict, assign=True, strict=False)
        model = model.to(device=device, dtype=torch.bfloat16)
    return model.eval()


def load_autoencoder(
    model_path: str,
    device: str | torch.device = "cuda"
) -> AutoEncoder:
    with torch.device("meta"):
        ae = AutoEncoder(configs["autoencoder"])
    ae.to_empty(device="cpu")
    state_dict = load_file(model_path, device="cpu")
    ae.load_state_dict(state_dict, assign=True, strict=False)
    ae = ae.to(device=device, dtype=torch.bfloat16)
    return ae.eval()


def load_t5(
    model_path: str,
    max_length: int = 512,
    device: str | torch.device = "cuda"
) -> HFEmbedder:
    with torch.device("meta"):
        t5 = HFEmbedder(
            model_path.parent,
            max_length=max_length,
            is_clip=False,
            dtype=torch.bfloat16
        )
    t5.to_empty(device="cpu")
    state_dict = load_file(model_path, device="cpu")
    # The checkpoint stores only hf_module.shared.weight for the tied token
    # embedding. Since transformers 5.x, encoder.embed_tokens is a separate
    # module (tied post-init), and assign=True breaks that tie — so alias the
    # key explicitly; both entries share one tensor, no extra memory.
    if "hf_module.shared.weight" in state_dict:
        state_dict.setdefault("hf_module.encoder.embed_tokens.weight", state_dict["hf_module.shared.weight"])
    t5.load_state_dict(state_dict, assign=True, strict=False)
    return t5.to(device=device, dtype=torch.bfloat16)


def load_t5_int4(
    model_path,
    tokenizer_dir,
    max_length: int = 512,
    device: str | torch.device = "cuda",
) -> HFEmbedder:
    """Load the nunchaku AWQ-INT4 T5 encoder (W4A16, ~3 GB vs 9 GB bf16).

    Expects the single-file checkpoint from nunchaku-ai/nunchaku-t5
    (awq-int4-flux.1-t5xxl.safetensors); its T5Config ships in the
    safetensors metadata, so only the tokenizer comes from tokenizer_dir.
    """
    from nunchaku import NunchakuT5EncoderModel  # optional dependency

    with torch.device("meta"):
        t5 = HFEmbedder(
            tokenizer_dir,
            max_length=max_length,
            is_clip=False,
            dtype=torch.bfloat16
        )
    t5.hf_module = NunchakuT5EncoderModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device=device
    ).eval()
    return t5


def load_clip(
    model_path: str,
    device: str | torch.device = "cuda"
) -> HFEmbedder:
    clip = HFEmbedder(
        model_path.parent,
        max_length=77,
        is_clip=True,
        dtype=torch.bfloat16
    )
    state_dict = load_file(model_path, device="cpu")
    state_dict = _remap_clip_state_dict(clip, state_dict)
    # Loading must be strict: silently retaining randomly initialized CLIP
    # parameters makes otherwise deterministic inference vary by process.
    clip.load_state_dict(state_dict, assign=True, strict=True)
    return clip.to(device=device, dtype=torch.bfloat16)
    
