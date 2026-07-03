"""CTC confidence / entropy signals (OPTIMIZATION_PLAN Phase 2 "Suggested Confidence Features").

Everything is derived from the encoder's BPE CTC logits plus the per-sample frame counts
``bpe_lengths``. Accepts either layout:

  packed flat ``[N_total, V]``  (the tensor ``_ctc_collapse_decode`` consumes), or
  dense padded ``[B, P, V]``    (the graph-break-free encoder output; pad rows are dropped
                                 by a tiny [B,P]-level gather, the [B,P,V] tensor is never copied)

We compute:

  frame-level  : top token, top prob, top log-prob, entropy, blank/non-blank
  token-level  : per *collapsed* token peak confidence, mean entropy, and the frame range that
                 produced it (used by the span-local editor to crop the audio, spans.py)
  per-sample   : mean/min token confidence, entropy mean/max, blank_ratio, repeat_token_ratio,
                 length_normalized_logprob, num_tokens

Invariant (unit-tested): the collapsed token ids produced here are *identical* to
``GraniteSpeechNarForASR._ctc_collapse_decode`` (unique_consecutive over argmax, drop blank).

Sync budget (P2.1): the whole batch costs ~5 host round-trips (2 nonzero counts, 2 tiny D2H
index transfers, 1 stacked-scalar D2H), replacing the previous ~11 syncs *per sample*.
Per-sample reduction order is preserved (reductions run on per-sample contiguous slices),
so every scalar is bit-identical to the per-sample implementation.

Entropy is in **nats** (natural log). Thresholds live in :class:`RoutingConfig` and MUST be
calibrated on the A100 profiler run (Phase 1 gate) — the defaults are conservative starting points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CTCConfidence:
    """Confidence bundle for a single utterance."""

    # per-sample scalars (python floats; ONE stacked GPU->CPU sync per batch materializes all of them)
    mean_token_confidence: float
    min_token_confidence: float
    entropy_mean: float
    entropy_max: float
    blank_ratio: float
    repeat_token_ratio: float
    length_normalized_logprob: float
    num_tokens: int

    # per-collapsed-token tensors (kept on-device; consumed by the span detector)
    token_ids: torch.Tensor = field(repr=False)          # [K] long  — the collapsed hypothesis
    token_confidence: torch.Tensor = field(repr=False)   # [K] float — peak posterior of each token
    token_entropy: torch.Tensor = field(repr=False)      # [K] float — mean frame entropy of each token
    token_frame_start: torch.Tensor = field(repr=False)  # [K] long  — first CTC frame of the token's run
    token_frame_end: torch.Tensor = field(repr=False)    # [K] long  — last+1 CTC frame of the token's run

    def as_dict(self) -> dict:
        """Scalar view (for logging / routing tables)."""
        return {
            "mean_token_confidence": self.mean_token_confidence,
            "min_token_confidence": self.min_token_confidence,
            "entropy_mean": self.entropy_mean,
            "entropy_max": self.entropy_max,
            "blank_ratio": self.blank_ratio,
            "repeat_token_ratio": self.repeat_token_ratio,
            "length_normalized_logprob": self.length_normalized_logprob,
            "num_tokens": self.num_tokens,
        }


def _empty_confidence(device, dtype=torch.long) -> CTCConfidence:
    z = torch.zeros(0, device=device)
    return CTCConfidence(
        mean_token_confidence=0.0, min_token_confidence=0.0,
        entropy_mean=0.0, entropy_max=0.0, blank_ratio=1.0,
        repeat_token_ratio=0.0, length_normalized_logprob=float("-inf"), num_tokens=0,
        token_ids=torch.zeros(0, dtype=torch.long, device=device),
        token_confidence=z, token_entropy=z,
        token_frame_start=torch.zeros(0, dtype=torch.long, device=device),
        token_frame_end=torch.zeros(0, dtype=torch.long, device=device),
    )


def compute_ctc_confidence(
    bpe_logits: torch.Tensor,
    bpe_lengths: list[int],
    blank_id: int,
) -> list[CTCConfidence]:
    """Confidence bundle per utterance (single-sync-class batched implementation).

    Args:
        bpe_logits:  ``[sum(bpe_lengths), V]`` packed flat logits, or ``[B, P, V]`` dense
                     padded logits (rows past each sample's length are ignored).
        bpe_lengths: per-sample frame counts (packed: sum == shape[0]; dense: len == B).
        blank_id:    CTC blank token id.
    """
    device = bpe_logits.device
    lengths = [int(l) for l in bpe_lengths]
    B = len(lengths)
    N = sum(lengths)
    if N == 0:
        return [_empty_confidence(device) for _ in range(B)]

    offsets = [0]
    for l in lengths:
        offsets.append(offsets[-1] + l)

    # ---- frame stats, computed per SAMPLE slice (not one flat launch): the CUDA softmax/
    # reduction launch strategy varies with row count, so per-sample slices are what keeps
    # every value BITWISE identical to the original per-sample implementation (verified);
    # it also caps the fp32 intermediates at one sample (~100 MB) instead of ~4 GB/batch.
    # No syncs here — just B small launches writing into preallocated flat outputs.
    if bpe_logits.dim() == 3:
        sample_rows = [bpe_logits[i, :lengths[i]] for i in range(B)]
    else:
        sample_rows = list(bpe_logits.split(lengths))
    top_logprob = torch.empty(N, dtype=torch.float32, device=device)
    top_idx = torch.empty(N, dtype=torch.long, device=device)
    entropy = torch.empty(N, dtype=torch.float32, device=device)
    # frame-level per-sample scalars are reduced HERE, on the standalone (offset-0) per-sample
    # tensors: CUDA reductions on views with a misaligned storage offset can differ by 1 ulp
    # (vectorized-load grouping), so reducing before the flat write keeps them bitwise equal
    # to the original per-sample implementation. Still 0 syncs — 0-dim GPU tensors, stacked later.
    frame_scal = [None] * B          # (ent_mean, ent_max, blank_ratio, lnl) per non-empty sample
    for i in range(B):
        if lengths[i] == 0:
            continue
        o0, o1 = offsets[i], offsets[i + 1]
        lp = torch.log_softmax(sample_rows[i].float(), dim=-1)
        tlp, tix = lp.max(dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1)
        top_logprob[o0:o1] = tlp
        top_idx[o0:o1] = tix
        entropy[o0:o1] = ent
        frame_scal[i] = (ent.mean(), ent.max(), (tix == blank_id).float().mean(), tlp.mean())
    top_prob = top_logprob.exp()

    # frame stats above are packed flat [N] for both layouts; sample-relative frame indices:
    offs_t = torch.as_tensor(offsets[:-1], dtype=torch.long, device=device)
    reps = torch.as_tensor(lengths, dtype=torch.long, device=device)
    off_per_frame = torch.repeat_interleave(offs_t, reps, output_size=N)
    local_frame = torch.arange(N, dtype=torch.long, device=device) - off_per_frame

    # ---- flat run segmentation with per-sample boundary force-keep: identical run boundaries
    # to per-sample ``unique_consecutive`` because a new run is forced at every sample start.
    change = torch.zeros(N, dtype=torch.bool, device=device)
    if N > 1:
        change[1:] = top_idx[1:] != top_idx[:-1]
    starts = [o for o, l in zip(offsets[:-1], lengths) if l > 0]
    change[torch.as_tensor(starts, dtype=torch.long, device=device)] = True
    run_id = torch.cumsum(change.to(torch.long), dim=0) - 1

    chg_pos = change.nonzero(as_tuple=False).flatten()            # sync 1 (data-dependent count)
    R = int(chg_pos.shape[0])
    run_vals = top_idx.index_select(0, chg_pos)

    # ---- per-run reductions (runs never cross sample boundaries by construction).
    ones = torch.ones(N, device=device)
    peak = torch.full((R,), float("-inf"), device=device).scatter_reduce(
        0, run_id, top_prob, reduce="amax", include_self=True)
    ent_sum = torch.zeros(R, device=device).scatter_add(0, run_id, entropy)
    cnt = torch.zeros(R, device=device).scatter_add(0, run_id, ones)
    run_entropy = ent_sum / cnt
    run_start = torch.full((R,), N, dtype=torch.long, device=device).scatter_reduce(
        0, run_id, local_frame, reduce="amin", include_self=True)
    run_end = torch.zeros(R, dtype=torch.long, device=device).scatter_reduce(
        0, run_id, local_frame + 1, reduce="amax", include_self=True)

    nonblank = run_vals != blank_id
    nb_idx = nonblank.nonzero(as_tuple=False).flatten()           # sync 2 (data-dependent count)
    token_ids_all = run_vals.index_select(0, nb_idx)
    token_conf_all = peak.index_select(0, nb_idx)
    token_ent_all = run_entropy.index_select(0, nb_idx)
    tok_fs_all = run_start.index_select(0, nb_idx)
    tok_fe_all = run_end.index_select(0, nb_idx)

    # ---- host bookkeeping: run->sample / token->sample counts from two tiny D2H transfers.
    chg_pos_host = chg_pos.cpu()                                  # sync 3 (R int64)
    nb_idx_host = nb_idx.cpu()                                    # sync 4 (K int64)
    offs_host = torch.as_tensor(offsets, dtype=torch.long)
    nb_pos_host = chg_pos_host.index_select(0, nb_idx_host)       # flat position of each token's run
    tok_bounds = torch.searchsorted(nb_pos_host, offs_host)
    k_counts = (tok_bounds[1:] - tok_bounds[:-1]).tolist()        # tokens per sample (host ints)

    token_ids_per = list(token_ids_all.split(k_counts))
    token_conf_per = list(token_conf_all.split(k_counts))
    token_ent_per = list(token_ent_all.split(k_counts))
    tok_fs_per = list(tok_fs_all.split(k_counts))
    tok_fe_per = list(tok_fe_all.split(k_counts))

    # ---- per-sample scalars as 0-dim GPU tensors (frame-level ones were reduced above on the
    # standalone per-sample tensors; token-level slices are clone()d to offset-0 storage first,
    # for the same 1-ulp alignment reason), stacked into one [B,7] D2H.
    zero = torch.zeros((), device=device)
    rows = []
    for i in range(B):
        if lengths[i] == 0:
            rows.append(torch.stack([zero, zero, zero, zero, zero, zero, zero]))
            continue
        e_mean, e_max, blank_ratio, lnl = frame_scal[i]
        if k_counts[i] == 0:
            rows.append(torch.stack([zero, zero, e_mean, e_max, blank_ratio, zero, lnl]))
        else:
            tc = token_conf_per[i].clone()
            if k_counts[i] > 1:
                rr = (token_ids_per[i][1:] == token_ids_per[i][:-1]).float().mean()
            else:
                rr = zero
            rows.append(torch.stack([tc.mean(), tc.min(), e_mean, e_max, blank_ratio, rr, lnl]))
    scal = torch.stack(rows).cpu()                                # sync 5 (ONE [B,7] D2H)

    out = []
    for i in range(B):
        if lengths[i] == 0:
            out.append(_empty_confidence(device))
            continue
        s = scal[i]
        K = k_counts[i]
        if K == 0:
            c = _empty_confidence(device)
            c.entropy_mean = float(s[2])
            c.entropy_max = float(s[3])
            c.blank_ratio = float(s[4])
            c.length_normalized_logprob = float(s[6])
            out.append(c)
            continue
        out.append(CTCConfidence(
            mean_token_confidence=float(s[0]),
            min_token_confidence=float(s[1]),
            entropy_mean=float(s[2]),
            entropy_max=float(s[3]),
            blank_ratio=float(s[4]),
            repeat_token_ratio=float(s[5]),
            length_normalized_logprob=float(s[6]),
            num_tokens=K,
            token_ids=token_ids_per[i],
            token_confidence=token_conf_per[i],
            token_entropy=token_ent_per[i],
            token_frame_start=tok_fs_per[i],
            token_frame_end=tok_fe_per[i],
        ))
    return out
