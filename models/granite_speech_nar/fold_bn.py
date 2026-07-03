"""Fold each Conformer conv module's BatchNorm1d into the preceding depthwise
Conv1d, in eval mode.

In eval, BatchNorm1d is a per-channel affine map  y = scale*z + shift  with
    scale = gamma / sqrt(var + eps),   shift = beta - mean*scale,
and the depthwise conv is linear, so  BN(conv(x)) = conv'(x)  with
    W' = W * scale[:, None, None],     b' = shift            (original bias = 0).
BN is then replaced by nn.Identity, removing one elementwise pass per block
(16 blocks). Standard NeMo-style inference fusion.

NOT bit-identical: the fold is computed in fp32 then cast back to the conv
weight dtype (bf16), and the fused conv+bias rounds differently from a separate
bf16 BN. Expect tiny deltas; transcript/argmax parity was validated against reference goldens
(and WER on a real set before accepting).

Usage (after load + .eval()):
    from models.granite_speech_nar.fold_bn import fold_conv_bn_
    model = load_model(REF, device="cuda")
    n = fold_conv_bn_(model)          # in-place, idempotent; returns #folded

Reference: NeMo Conformer folds BN into the depthwise conv weights at load
(https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html).
"""
from __future__ import annotations

import torch
from torch import nn

from .conformer import ConformerConvModule


@torch.no_grad()
def fold_conv_bn_(model: nn.Module) -> int:
    """In-place fold of BatchNorm1d into the depthwise Conv1d of every
    ConformerConvModule. Returns the number of modules folded. Idempotent."""
    if model.training:
        raise RuntimeError("fold_conv_bn_ requires eval mode (call model.eval() first)")

    folded = 0
    for m in model.modules():
        if not isinstance(m, ConformerConvModule):
            continue
        if isinstance(m.batch_norm, nn.Identity):
            continue  # already folded

        conv = m.depth_conv.conv          # nn.Conv1d, depthwise (groups=C), bias=False
        bn = m.batch_norm                 # nn.BatchNorm1d (eval, running stats)
        if bn.running_mean is None or bn.running_var is None:
            raise RuntimeError("BatchNorm has no running stats; cannot fold")

        w_dtype = conv.weight.dtype
        w = conv.weight.detach().float()                       # (C, 1, K)
        gamma = bn.weight.detach().float()                     # (C,)
        beta = bn.bias.detach().float()
        mean = bn.running_mean.detach().float()
        var = bn.running_var.detach().float()
        eps = float(bn.eps)

        scale = gamma * torch.rsqrt(var + eps)                 # (C,)
        new_w = w * scale.reshape(-1, 1, 1)
        new_b = beta - mean * scale                            # conv had no bias

        conv.weight = nn.Parameter(new_w.to(w_dtype), requires_grad=False)
        conv.bias = nn.Parameter(new_b.to(w_dtype), requires_grad=False)
        m.batch_norm = nn.Identity()
        folded += 1

    return folded
