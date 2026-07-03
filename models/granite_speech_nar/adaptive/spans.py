"""Suspicious-span detection + windowing + merge-back (OPTIMIZATION_PLAN Phase 3).

Works in *collapsed-token index space* (the CTC hypothesis after blank removal). A span is a
half-open ``(start, end)`` range of token indices.

Flow:
  detect_suspicious_spans(conf, cfg)          -> raw risky spans (low conf / high entropy / repeat)
  window_spans(spans, n_tokens, L, R)         -> merged spans padded with L/R context tokens
  span_audio_frame_range(...)                 -> map a token window to an audio-embed slice (crop)
  merge_local_edits(ctc_tokens, windows, eds) -> splice edited windows back into the full hypothesis

The audio-frame mapping is approximate (CTC pooled-frame rate != projector downsample rate); we add
a margin and clamp, and lean on the L/R token context, so the editor still sees enough acoustics.
The *index arithmetic* is exact and unit-tested; the *accuracy* of local editing must be validated
on the A100 WER suite before loosening thresholds.
"""
from __future__ import annotations

import torch


def _risky_mask(conf, cfg) -> torch.Tensor | None:
    """On-device risky mask for one sample's collapsed tokens (no host syncs)."""
    k = int(conf.num_tokens)
    if k == 0:
        return None
    tconf = conf.token_confidence
    tent = conf.token_entropy
    tids = conf.token_ids

    risky = (tconf < cfg.span_conf_threshold) | (tent > cfg.span_entropy_threshold)
    if k > 1:
        rep = torch.zeros(k, dtype=torch.bool, device=tids.device)
        eq = tids[1:] == tids[:-1]
        rep[1:] |= eq
        rep[:-1] |= eq
        risky = risky | rep
    return risky


def _group_risky(risky_idx: list[int], merge_gap: int) -> list[tuple[int, int]]:
    """Group consecutive risky indices, merging gaps <= merge_gap (host ints, no tensors)."""
    if not risky_idx:
        return []
    spans: list[list[int]] = []
    start = prev = risky_idx[0]
    for i in risky_idx[1:]:
        if i - prev <= merge_gap:
            prev = i
        else:
            spans.append([start, prev + 1])
            start = prev = i
    spans.append([start, prev + 1])
    return [(s, e) for s, e in spans]


def detect_suspicious_spans(conf, cfg) -> list[tuple[int, int]]:
    """Return merged half-open ``(start, end)`` token spans flagged as risky.

    A token is risky if its confidence < ``cfg.span_conf_threshold``, its entropy >
    ``cfg.span_entropy_threshold``, or it is part of a consecutive repetition. Adjacent risky
    tokens (within ``cfg.span_merge_gap``) are merged into one span.
    """
    risky = _risky_mask(conf, cfg)
    if risky is None:
        return []
    risky_idx = torch.nonzero(risky, as_tuple=False).flatten().tolist()
    return _group_risky(risky_idx, cfg.span_merge_gap)


def detect_suspicious_spans_batch(confs, cfg) -> list[list[tuple[int, int]]]:
    """Batched :func:`detect_suspicious_spans`: identical results, ONE host sync for the
    whole batch (P2.1b) instead of one ``nonzero().tolist()`` per sample. Masks are built
    on-device, concatenated, moved with a single ``.cpu()``, and grouped on host."""
    masks = [_risky_mask(c, cfg) for c in confs]
    real = [m for m in masks if m is not None]
    if not real:
        return [[] for _ in confs]
    flat = torch.cat(real).cpu()                       # ONE sync for the whole batch
    out: list[list[tuple[int, int]]] = []
    off = 0
    for m in masks:
        if m is None:
            out.append([])
            continue
        k = m.shape[0]
        idxs = flat[off:off + k].nonzero(as_tuple=False).flatten().tolist()
        off += k
        out.append(_group_risky(idxs, cfg.span_merge_gap))
    return out


def window_spans(spans: list[tuple[int, int]], n_tokens: int, left: int, right: int) -> list[tuple[int, int]]:
    """Add L/R context tokens to each span, clamp to [0, n_tokens], and merge overlaps."""
    if not spans:
        return []
    padded = sorted(
        (max(0, s - left), min(n_tokens, e + right)) for s, e in spans
    )
    merged = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= merged[-1][1]:            # overlapping or touching -> merge
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def span_audio_frame_range(
    frame_start: int, frame_end: int, pool_window: int, downsample_rate: int,
    audio_len: int, margin: int = 4,
) -> tuple[int, int]:
    """Map a CTC (pooled) frame range to a half-open audio-embed index range, with margin+clamp.

    CTC logits live at the pooled rate (encoder_frames / pool_window); audio embeddings live at
    encoder_frames / downsample_rate. So audio_idx ~= ctc_frame * pool_window / downsample_rate.
    """
    scale = pool_window / downsample_rate
    a0 = int(frame_start * scale) - margin
    a1 = int(-(-frame_end * scale // 1)) + margin  # ceil(frame_end*scale) + margin
    # cap a0 below audio_len (not just >= 0) so a0 < a1 even when scale >= 1 or margin == 0
    a0 = max(0, min(a0, max(0, audio_len - 1)))
    a1 = min(audio_len, max(a1, a0 + 1))
    return a0, a1


def merge_local_edits(
    ctc_tokens: torch.Tensor,
    windows: list[tuple[int, int]],
    edited_windows: list[torch.Tensor],
) -> torch.Tensor:
    """Splice each edited window back into the full CTC hypothesis.

    ``windows`` must be sorted and non-overlapping (as produced by :func:`window_spans`).
    ``edited_windows[i]`` replaces ``ctc_tokens[windows[i][0]:windows[i][1]]``.
    """
    if not windows:
        return ctc_tokens
    parts: list[torch.Tensor] = []
    cursor = 0
    for (ws, we), ed in zip(windows, edited_windows):
        if ws > cursor:
            parts.append(ctc_tokens[cursor:ws])
        parts.append(ed.to(ctc_tokens.dtype))
        cursor = we
    if cursor < ctc_tokens.shape[0]:
        parts.append(ctc_tokens[cursor:])
    if not parts:
        return ctc_tokens[:0]
    return torch.cat(parts, dim=0)
