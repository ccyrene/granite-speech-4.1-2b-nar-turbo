"""Build the pure-torch model and load checkpoint weights (no transformers).

Parameter names mirror the checkpoint exactly, so loading is an identity-keyed
``load_state_dict(strict=True)``. We use ``assign=True`` so each tensor keeps its
stored dtype (bf16) while non-checkpoint buffers (RoPE ``inv_freq``) keep their
float32 init — matching the reference's mixed-precision behaviour.
"""
from __future__ import annotations

import os

import torch
from safetensors.torch import load_file

from .config import GraniteSpeechNarConfig
from .asr import GraniteSpeechNarForASR


def load_state_dict_from_safetensors(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    return load_file(path)


def build_model(config_path: str) -> GraniteSpeechNarForASR:
    config = GraniteSpeechNarConfig.from_json_file(config_path)
    return GraniteSpeechNarForASR(config)


def load_model(model_dir: str, device: str = "cpu") -> GraniteSpeechNarForASR:
    """Load a ready-to-run model from a directory holding config.json + model.safetensors."""
    config_path = os.path.join(model_dir, "config.json")
    model = build_model(config_path)

    sd = load_state_dict_from_safetensors(model_dir)
    missing, unexpected = model.load_state_dict(sd, strict=True, assign=True)
    if missing or unexpected:  # strict=True already raises, but keep for clarity
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")

    model.eval()
    model.to(device)
    if hasattr(model.encoder, "cache_rel_pos_tables"):
        model.encoder.cache_rel_pos_tables()   # P2.4b: constant Shaw tables, gathered once
    from . import attn_backends
    ck = attn_backends.conv_kernel()
    if ck in ("dwconv_silu", "dwconv_silu_glu"):
        from .conv_kernel import enable_dwconv_silu
        n = enable_dwconv_silu(model, mode=ck)
        print(f"[loader] GRANITE_CONV_KERNEL={ck} -> {n} conv modules fused")
    return model
