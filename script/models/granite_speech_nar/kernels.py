"""Opt-in custom-kernel backend for the LLM editor (OPTIMIZATION_PLAN Phase 6 integration).

Default: **OFF** — the bit-exact torch path in ``llm.py`` runs unchanged, so the lossless baseline
is untouched. Enable with ``GRANITE_KERNEL_BACKEND=cuda`` (or ``cutile``); it only takes effect on
CUDA tensors, and only if ``cuda_kernels`` + its NVRTC/cuTile toolchain import successfully —
otherwise it silently falls back to the torch reference.

The winning fused kernels (rmsnorm, silu_mul, rope, add_scale) are bit-exact vs torch on the dev
card (see cuda_kernels/README); they must still be END-TO-END WER-validated on the A100 before being
turned on in production. Only ``rmsnorm`` and ``silu_mul`` are wired into ``llm.py`` so far (the two
highest-value fusions whose reference math matches exactly); rope/add_scale remain available in
``cuda_kernels`` but unwired to keep the un-validatable surface small.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_BACKEND = os.environ.get("GRANITE_KERNEL_BACKEND", "").strip().lower()
_MOD = None
_TRIED = False


def backend() -> str:
    return _BACKEND


def enabled() -> bool:
    return _BACKEND in ("cuda", "cutile")


def _dispatch():
    """Lazily import the cuda_kernels dispatcher; None if unavailable (e.g. CPU box, no NVRTC)."""
    global _MOD, _TRIED
    if _MOD is None and not _TRIED:
        _TRIED = True
        try:
            from cuda_kernels import best as _best
            _MOD = _best
        except Exception:
            _MOD = None
    return _MOD


# --------------------------------------------------------------------------- #
# Bit-exact torch references (always available; CPU-testable). These MUST match
# the inline math in llm.py so that enabling the backend is value-preserving.
# --------------------------------------------------------------------------- #
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def silu_mul_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


# --------------------------------------------------------------------------- #
# Dispatch: kernel when enabled+cuda+available, else the reference.
# --------------------------------------------------------------------------- #
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if enabled() and x.is_cuda:
        m = _dispatch()
        fn = getattr(m, "rmsnorm", None) if m is not None else None
        if fn is not None:
            try:
                return fn(x, weight, eps)
            except Exception:
                pass
    return rmsnorm_ref(x, weight, eps)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if enabled() and gate.is_cuda:
        m = _dispatch()
        fn = getattr(m, "silu_mul", None) if m is not None else None
        if fn is not None:
            try:
                return fn(gate, up)
            except Exception:
                pass
    return silu_mul_ref(gate, up)
