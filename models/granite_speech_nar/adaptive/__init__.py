"""Adaptive-inference layer for Granite Speech NAR (OPTIMIZATION_PLAN Phases 2-8).

Everything here is *additive* and *opt-in*: the default ``model.transcribe(...)`` path
is untouched (bit-exact lossless baseline). The adaptive path is reached only through
``model.transcribe_adaptive(...)`` with an :class:`AdaptiveConfig` whose ``enabled`` flag
defaults to ``False``.

Pipeline the adaptive layer implements on top of the encoder+CTC output:

    encoder -> CTC logits
      -> compute_ctc_confidence            (confidence.py, Phase 2 features)
      -> route_decision                    (routing.py,   Phase 2 gate)
           A. ctc_fast : return the CTC hypothesis directly (skip the editor)
           B. local    : edit only suspicious windows      (spans.py, Phase 3)
                         + verifier_accepts, else fall back (verifier.py, Phase 4 cascade)
           C. full     : run the full NAR editor           (unchanged path)

All the pure-tensor pieces (confidence, routing, spans, packing) are unit-tested on
CPU with synthetic logits in ``scripts/test_adaptive.py`` — no GPU / weights needed.
"""
from __future__ import annotations

from .confidence import CTCConfidence, compute_ctc_confidence
from .routing import RoutingConfig, Route, route_decision
from .verifier import VerifierConfig, verifier_accepts
from .spans import (
    detect_suspicious_spans,
    detect_suspicious_spans_batch,
    window_spans,
    span_audio_frame_range,
    merge_local_edits,
)
from .packing import length_aware_batches, padding_ratio, avg_padding_ratio
from .early_exit import EarlyExitConfig, stable_argmax
from .editor_pack import (
    interleave_blank, build_packed_segments, split_text_hidden, text_segment_lengths,
    text_position_mask,
)
from .ctc_fused import ctc_backend, fused_ctc_features
from .config_io import (
    routing_from_dict, verifier_from_dict, early_exit_from_dict, load_adaptive_config,
)

__all__ = [
    "CTCConfidence",
    "compute_ctc_confidence",
    "RoutingConfig",
    "Route",
    "route_decision",
    "VerifierConfig",
    "verifier_accepts",
    "detect_suspicious_spans",
    "detect_suspicious_spans_batch",
    "window_spans",
    "span_audio_frame_range",
    "merge_local_edits",
    "length_aware_batches",
    "padding_ratio",
    "avg_padding_ratio",
    "EarlyExitConfig",
    "stable_argmax",
    "interleave_blank",
    "build_packed_segments",
    "split_text_hidden",
    "text_segment_lengths",
    "text_position_mask",
    "ctc_backend",
    "fused_ctc_features",
    "routing_from_dict",
    "verifier_from_dict",
    "early_exit_from_dict",
    "load_adaptive_config",
]
