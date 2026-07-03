"""Fused depthwise Conv1d + bias + SiLU custom op for the Conformer conv module.

After ``fold_conv_bn_`` the conv module is
``silu(depthwise_conv(x) + bias)`` — one hand-NVRTC kernel (cuda_kernels/conv1d.py)
replaces the conv + BN + SiLU chain. Registered as a ``torch.library`` custom op
(``granite::dwconv_silu``) so a compiled encoder sees ONE opaque node: no graph break,
fullgraph-safe, launches on torch's current stream.

Opt-in: ``GRANITE_CONV_KERNEL=dwconv_silu`` (read in loader.load_model, or apply the
``convkernel`` lever in scripts/bench_asr.py). ``enable_dwconv_silu(model)`` folds BN,
test-fires the kernel against the torch path and only then flips each module's
``_fused_conv`` flag — any toolchain/launch failure falls back to the torch path.

NOT bit-identical (the BN fold itself is fp32-computed then bf16-cast) — WER-gated like
``encattn``; transcript/argmax parity was validated against reference-implementation goldens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_READY = False
_ERR = ""


def _register() -> bool:
    """Register granite::dwconv_silu + granite::dwconv_silu_glu once; False (with reason
    in _ERR) if the NVRTC toolchain is unavailable. PTX compile is deferred to first call."""
    global _READY, _ERR
    if _READY:
        return True
    if _ERR:
        return False
    try:
        from cuda_kernels.conv1d import dwconv1d_silu as _impl
        from cuda_kernels.conv1d import dwconv1d_silu_glu as _impl_glu
    except Exception as e:  # no cuda-python / NVRTC headers on this box
        _ERR = f"cuda_kernels import failed: {e}"
        return False

    @torch.library.custom_op("granite::dwconv_silu", mutates_args=(), device_types="cuda")
    def dwconv_silu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return _impl(x, weight, bias)

    @dwconv_silu.register_fake
    def _(x, weight, bias):
        return x.new_empty(x.shape)

    @torch.library.custom_op("granite::dwconv_silu_glu", mutates_args=(), device_types="cuda")
    def dwconv_silu_glu(x2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return _impl_glu(x2, weight, bias)

    @dwconv_silu_glu.register_fake
    def _(x2, weight, bias):
        b, c2, t = x2.shape
        return x2.new_empty((b, c2 // 2, t))

    _READY = True
    return True


@torch.no_grad()
def enable_dwconv_silu(model: torch.nn.Module, mode: str = "dwconv_silu") -> int:
    """Fold BN into the depthwise convs and switch every eligible ConformerConvModule to
    the fused kernel. ``mode``: "dwconv_silu" (conv+bias+SiLU) or "dwconv_silu_glu"
    (GLU folded in too — the (B,C,T) GLU intermediate never materializes).
    Returns the number of modules switched (0 = torch-path fallback)."""
    from .conformer import ConformerConvModule
    from .fold_bn import fold_conv_bn_

    if mode not in ("dwconv_silu", "dwconv_silu_glu"):
        print(f"[conv_kernel] fallback to torch path: unknown mode {mode!r}")
        return 0
    if not _register():
        print(f"[conv_kernel] fallback to torch path: {_ERR}")
        return 0

    mods = [m for m in model.modules() if isinstance(m, ConformerConvModule)]
    if not mods:
        return 0
    fold_conv_bn_(model)

    conv0 = mods[0].depth_conv.conv
    dev, dtype = conv0.weight.device, conv0.weight.dtype
    if dev.type != "cuda" or dtype != torch.bfloat16:
        print(f"[conv_kernel] fallback to torch path: needs CUDA bf16 (got {dev.type}/{dtype})")
        return 0
    try:  # test-fire: catches NVRTC compile/launch failures and value blowups up front
        C = conv0.weight.shape[0]
        if mode == "dwconv_silu_glu":
            x2 = torch.randn(2, 2 * C, 37, device=dev, dtype=dtype)
            ref = F.silu(conv0(F.glu(x2, dim=1)))
            out = torch.ops.granite.dwconv_silu_glu(x2, conv0.weight, conv0.bias)
        else:
            x = torch.randn(2, C, 37, device=dev, dtype=dtype)
            ref = F.silu(conv0(x))  # post-fold reference: padding folded into conv, bias present
            out = torch.ops.granite.dwconv_silu(x, conv0.weight, conv0.bias)
        err = (out.float() - ref.float()).abs().max().item()
        # bf16 ulp scales with magnitude: with real folded weights |ref| can reach ~16, where
        # 1 ulp = 6.25e-2 — gate on a magnitude-relative bound (~2 ulp), not an absolute one.
        tol = 2e-2 * max(1.0, ref.float().abs().max().item())
        if err > tol:
            print(f"[conv_kernel] fallback to torch path: test-fire max|Δ|={err:.3e} > tol {tol:.3e}")
            return 0
    except Exception as e:
        print(f"[conv_kernel] fallback to torch path: test-fire failed: {e}")
        return 0

    n = 0
    for m in mods:
        conv = m.depth_conv.conv
        if (isinstance(m.batch_norm, torch.nn.Identity) and conv.bias is not None
                and m.depth_conv._symmetric):
            m._fused_conv = mode
            n += 1
    return n
