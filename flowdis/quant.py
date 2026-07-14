"""INT8 ConvRot quantized inference support.

Loads checkpoints produced by convert_to_quant (ctq) in the ComfyUI-native
int8 format: row-wise INT8 weights, optionally pre-rotated with a group-wise
regular Hadamard transform (ConvRot, https://arxiv.org/abs/2512.03673).

Per quantized layer the checkpoint stores:
  <name>.weight        int8, (out, in), already rotated when convrot is on
  <name>.weight_scale  float32, (out, 1) row-wise dequant scale
  <name>.comfy_quant   uint8 tensor holding a JSON config, e.g.
                       {"format": "int8_tensorwise", "per_row": true,
                        "convrot": true, "convrot_groupsize": 256}

At inference the activation is rotated with the same block-diagonal Hadamard
(orthogonal, so the product is unchanged), dynamically quantized per row to
int8, multiplied with torch._int_mm, and rescaled with the outer product of
the two scales. When comfy-kitchen is installed its fused CUDA kernel does
all four steps in one launch and is used instead.
"""

import json
import os

import torch
from torch import Tensor, nn

# comfy-kitchen ships a fully fused CUDA kernel (rotation + dynamic quant +
# int8 GEMM + rescale in one launch) that is ~1.4x faster than our
# torch.compile path at real token counts and has no M>16 restriction.
# Optional dependency; opt out with FLOWDIS_INT8_KITCHEN=0.
if os.environ.get("FLOWDIS_INT8_KITCHEN", "1") != "0":
    try:
        import comfy_kitchen as _kitchen
    except ImportError:
        _kitchen = None
else:
    _kitchen = None

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


def _int8_gemm_rowwise(x2d: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, out_dtype: torch.dtype) -> Tensor:
    """Dynamic per-row activation quant + int8 GEMM + rescale.

    Kept as a standalone function so torch.compile can fuse the quantization
    and rescale chains around torch._int_mm; eager, these memory-bound passes
    cost more than the GEMM itself (~75% of layer time).
    """
    m = x2d.shape[0]
    x_scale = x2d.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8) / 127.0
    xq = torch.round(x2d.float() / x_scale).clamp_(-127, 127).to(torch.int8)
    if m <= 16:  # torch._int_mm requires m > 16
        xq = torch.cat([xq, xq.new_zeros(17 - m, xq.shape[1])])
    y = torch._int_mm(xq, weight.t())[:m].float() * (x_scale * weight_scale.float().t())
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype)


if os.environ.get("FLOWDIS_INT8_COMPILE", "1") != "0":
    # The function is tiny, so one compile thread is plenty; inductor's default
    # forked worker pool can hang on CUDA-initialized parents.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    _int8_gemm_rowwise_opt = torch.compile(_int8_gemm_rowwise, dynamic=True)
else:
    _int8_gemm_rowwise_opt = _int8_gemm_rowwise


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Normalized regular Hadamard matrix built as Kronecker powers of H4.

    The regular H4 (every row/column sums to 2) avoids the all-ones column of
    Sylvester matrices, which would amplify row-wise outliers. Size must be a
    power of 4, matching the group sizes ctq accepts.
    """
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    k = size.bit_length() - 1
    if size < 4 or (1 << k) != size or k % 2 != 0:
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")

    H4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float64,
        device=device,
    )
    H = H4
    while H.shape[0] < size:
        H = torch.kron(H, H4)
    H = (H / (size**0.5)).to(dtype)
    _HADAMARD_CACHE[cache_key] = H
    return H


def rotate_activation(x: Tensor, H: Tensor, group_size: int) -> Tensor:
    """Block-diagonal rotation x @ diag(H, ..., H), applied group-wise."""
    orig_shape = x.shape
    x = x.reshape(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)
    x = torch.matmul(x, H.to(dtype=x.dtype, device=x.device))
    return x.reshape(orig_shape)


class Int8ConvRotLinear(nn.Module):
    """Drop-in replacement for nn.Linear backed by a row-wise INT8 weight.

    W8A8 path: per-row dynamic activation quantization + torch._int_mm.
    torch._int_mm requires the flattened token count to exceed 16; shorter
    inputs are padded up to 17 rows (zero rows are free wrt correctness).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        convrot_groupsize: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = convrot_groupsize
        self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros(out_features, 1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, convrot_groupsize={self.convrot_groupsize}"
        )

    def forward(self, x: Tensor) -> Tensor:
        out_dtype = x.dtype
        batch_shape = x.shape[:-1]

        if x.device.type == "cuda" and _kitchen is not None:
            # Fused kernel does the activation rotation internally.
            y = _kitchen.int8_linear(
                x.reshape(-1, self.in_features).contiguous(),
                self.weight,
                self.weight_scale,
                bias=self.bias,
                out_dtype=out_dtype,
                convrot=self.convrot_groupsize is not None,
                convrot_groupsize=self.convrot_groupsize or 256,
            )
            return y.reshape(*batch_shape, self.out_features)

        if self.convrot_groupsize is not None:
            H = build_hadamard(self.convrot_groupsize, device=x.device, dtype=x.dtype)
            x = rotate_activation(x, H, self.convrot_groupsize)

        x2d = x.reshape(-1, self.in_features)

        if x.device.type == "cuda":
            y = _int8_gemm_rowwise_opt(x2d, self.weight, self.weight_scale, self.bias, out_dtype)
        else:
            w = (self.weight.float() * self.weight_scale.float()).to(out_dtype)
            y = x2d @ w.t()
            if self.bias is not None:
                y = y + self.bias.to(out_dtype)
        return y.reshape(*batch_shape, self.out_features)


def _decode_comfy_quant(t: Tensor) -> dict:
    return json.loads(t.cpu().numpy().tobytes().decode("utf-8"))


def swap_quantized_linears(model: nn.Module, state_dict: dict[str, Tensor]) -> int:
    """Replace nn.Linear modules with Int8ConvRotLinear for every layer that
    carries a `.comfy_quant` config in the checkpoint. Returns the swap count.
    """
    n_swapped = 0
    for key in list(state_dict.keys()):
        if not key.endswith(".comfy_quant"):
            continue
        name = key[: -len(".comfy_quant")]
        config = _decode_comfy_quant(state_dict.pop(key))

        fmt = config.get("format")
        if fmt != "int8_tensorwise" or not config.get("per_row"):
            raise ValueError(f"Unsupported quantization config for {name}: {config}")

        parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
        child_name = name.rpartition(".")[2]
        old = getattr(parent, child_name)
        if not isinstance(old, nn.Linear):
            raise ValueError(f"{name} has quantized weights but is {type(old).__name__}")

        groupsize = config.get("convrot_groupsize") if config.get("convrot") else None
        new = Int8ConvRotLinear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            convrot_groupsize=groupsize,
        )
        setattr(parent, child_name, new)
        n_swapped += 1

        # Scales may be stored as scalars or flat vectors; normalize to (out, 1).
        scale = state_dict[f"{name}.weight_scale"]
        state_dict[f"{name}.weight_scale"] = scale.reshape(-1, 1).expand(old.out_features, 1).contiguous().float()

    return n_swapped


def is_quantized_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(k.endswith(".comfy_quant") for k in state_dict)
