"""Route-decision logic (OPTIMIZATION_PLAN Phase 2 "Route Decision Logic").

Given the CTC confidence bundle + detected suspicious spans, decide per utterance:

    ctc_fast : CTC is reliable   -> return CTC hypothesis, skip the editor entirely
    local    : a few small risky spans -> edit only those windows (Phase 3 / cascade tier-1)
    full     : noisy / long / uncertain -> run the full NAR editor (unchanged path)

Policy is intentionally conservative (skip the editor only when confidence is *extremely* high),
per the plan's "start conservative, then loosen after evaluation". Every threshold is a config
knob to be tuned against the Phase-1 profiler + accuracy suite; nothing here should regress WER
by default because ``AdaptiveConfig.enabled`` is False unless explicitly turned on.
"""
from __future__ import annotations

from dataclasses import dataclass


class Route:
    CTC_FAST = "ctc_fast"
    LOCAL = "local"
    FULL = "full"


@dataclass
class RoutingConfig:
    """Thresholds for the CTC-first router. Units: confidence in [0,1], entropy in nats."""

    enabled: bool = False

    # ---- ctc_fast gate (must ALL pass to skip the editor) ---------------------------- #
    high_conf_threshold: float = 0.92        # mean_token_confidence >=
    min_token_conf_threshold: float = 0.75   # min_token_confidence  >=
    max_entropy: float = 0.45                # entropy_mean          <=
    max_blank_ratio: float = 0.90            # blank_ratio           <=  (all-blank => degenerate)
    max_repeat_ratio: float = 0.30           # repeat_token_ratio    <=

    # ---- local vs full ---------------------------------------------------------------- #
    max_local_spans: int = 3                 # suspicious_span_count <=
    max_local_tokens: int = 12               # total suspicious tokens (pre-window) <=

    # ---- span detection (consumed by spans.detect_suspicious_spans) ------------------- #
    span_conf_threshold: float = 0.60        # a token is suspicious if confidence <  this
    span_entropy_threshold: float = 1.50     # ...or entropy > this (nats)
    span_merge_gap: int = 2                  # merge suspicious runs separated by <= this many tokens
    left_context_tokens: int = 12            # window context added on each side (Phase 3 windowing rule)
    right_context_tokens: int = 12

    # ---- local editor mode ------------------------------------------------------------ #
    # When False (default, SAFE): a "local" route runs the FULL editor (local == full); the only
    # behavioural change from the baseline is the ctc_fast skip. When True: the compute-saving
    # cropped-audio windowed editor is used (Phase 3 proper) — MUST be A100 WER-validated first.
    local_crop_audio: bool = False
    audio_crop_margin: int = 4               # extra audio-embed frames kept on each side of a window

    # ---- cascade / verifier ----------------------------------------------------------- #
    fallback_to_full_editor: bool = True     # if the local edit is not verified, run the full editor

    def __post_init__(self):
        if not (0.0 <= self.high_conf_threshold <= 1.0):
            raise ValueError("high_conf_threshold must be in [0,1]")
        if self.max_local_spans < 0 or self.max_local_tokens < 0:
            raise ValueError("max_local_* must be >= 0")


def route_decision(conf, spans: list[tuple[int, int]], cfg: RoutingConfig) -> str:
    """Return one of ``Route.{CTC_FAST,LOCAL,FULL}`` for one utterance.

    Args:
        conf:  a :class:`CTCConfidence` bundle for the utterance.
        spans: suspicious ``(start, end)`` token spans (pre-window) from
               :func:`spans.detect_suspicious_spans`.
        cfg:   :class:`RoutingConfig`.
    """
    n_spans = len(spans)
    n_suspect_tokens = sum(e - s for s, e in spans)

    # A) high-confidence fast path — skip the editor
    if (
        n_spans == 0
        and conf.num_tokens > 0
        and conf.mean_token_confidence >= cfg.high_conf_threshold
        and conf.min_token_confidence >= cfg.min_token_conf_threshold
        and conf.entropy_mean <= cfg.max_entropy
        and conf.blank_ratio <= cfg.max_blank_ratio
        and conf.repeat_token_ratio <= cfg.max_repeat_ratio
    ):
        return Route.CTC_FAST

    # B) a few small risky spans -> local editor
    if (
        0 < n_spans <= cfg.max_local_spans
        and n_suspect_tokens <= cfg.max_local_tokens
    ):
        return Route.LOCAL

    # C) everything else -> full editor
    return Route.FULL
