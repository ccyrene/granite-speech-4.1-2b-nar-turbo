"""CTC-postprocess fusion (OPTIMIZATION_PLAN Phase 6, Kernels 1-3).

The plan proposes fusing the CTC postprocess into custom kernels:
  Kernel 1: argmax + token confidence + entropy      (per frame)
  Kernel 2: collapse + blank removal                  (-> collapsed tokens + confidence + offsets)
  Kernel 3: suspicious-span detection                 (-> spans + route decision)

That fusion is already expressed as a *single torch pass* in ``confidence.py`` (one softmax +
run-length segmentation + scatter-reduce -> Kernels 1&2) and ``spans.py`` (Kernel 3). This module is
the integration point that a Triton kernel would slot behind: identical inputs/outputs, selected by
``GRANITE_CTC_BACKEND``. The torch path is the correct reference and the default; a Triton port is a
drop-in future optimization (needs a GPU + triton to author/validate — absent on the authoring box).
"""
from __future__ import annotations

import os

from .confidence import compute_ctc_confidence
from .spans import detect_suspicious_spans


def ctc_backend() -> str:
    return os.environ.get("GRANITE_CTC_BACKEND", "torch").strip().lower()


def fused_ctc_features(bpe_logits_flat, bpe_lengths, blank_id: int, routing):
    """Kernels 1-3 fused. Returns ``[(CTCConfidence, [(start,end), ...]), ...]`` per sample.

    ``GRANITE_CTC_BACKEND=triton`` is reserved for a future kernel; today it always uses the torch
    reference (which is the fused single-pass computation the kernel would replicate).
    """
    confs = compute_ctc_confidence(bpe_logits_flat, bpe_lengths, blank_id)
    return [(c, detect_suspicious_spans(c, routing)) for c in confs]
