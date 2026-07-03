"""Attention masks for the bidirectional, packed LLM (pure torch).

The ASR model packs multiple samples into one length dimension as
``[audio_0; text_0; audio_1; text_1; ...]`` with ``position_ids`` resetting to 0
at each sample boundary. Attention must be:
  * fully bidirectional *within* a sample, and
  * blocked *across* samples.

For a single sample (batch=1) every position shares one segment, so the mask is
all-ones and we return ``None`` (full attention) to match the reference's
fast path exactly.
"""
from __future__ import annotations

import torch


def packed_segment_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Assign a segment id to each position; a new segment starts where position==0."""
    starts = (position_ids == 0).to(torch.long)
    return starts.cumsum(dim=-1)


def build_bidirectional_mask(position_ids: torch.Tensor) -> torch.Tensor | None:
    """Return an additive float mask (1,1,L,L) or ``None`` if a single segment.

    ``None`` => unrestricted bidirectional attention (the batch=1 fast path).
    """
    seg = packed_segment_ids(position_ids)  # (B, L)
    if seg.max() <= 1:
        return None
    # same-segment => allowed
    allow = seg.unsqueeze(-1) == seg.unsqueeze(-2)  # (B, L, L)
    return allow.unsqueeze(1)  # (B, 1, L, L) boolean (True = attend)
