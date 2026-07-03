"""Packed-segment construction for the NAR editor (shared by the full + local-window routes).

The editor consumes one flat length dimension holding ``[audio_0; text_0; audio_1; text_1; ...]``
with ``position_ids`` resetting to 0 per segment (``masking.build_bidirectional_mask`` then blocks
cross-segment attention). This generalizes ``asr._build_flat_inputs``:

  * full route  : one segment per sample = (full audio embeds, full CTC hypothesis)
  * local route : one segment per suspicious window = (cropped audio embeds, window tokens)

Keeping the packing/splitting here (pure index bookkeeping) lets it be unit-tested on CPU with a
stub embedding — the off-by-one-prone part is validated without weights.
"""
from __future__ import annotations

import torch


def interleave_blank(token_ids: torch.Tensor, blank_id: int, min_len: int = 8) -> torch.Tensor:
    """Insert blank editing slots: (blank, tok, blank, tok, ..., blank).

    Identical to ``GraniteSpeechNarForASR._add_insertion_slots`` (kept in sync by test).
    """
    n = int(token_ids.numel())
    total_len = max(2 * n + 1, min_len)
    out = torch.full((total_len,), blank_id, dtype=token_ids.dtype, device=token_ids.device)
    if n > 0:
        idx = torch.arange(n, device=token_ids.device)
        out[2 * idx + 1] = token_ids
    return out


def build_packed_segments(segments, embed_tokens, blank_id: int, min_edit_len: int = 8,
                          seq_grid: int = 0):
    """Pack ``[(audio_embed[La,D], text_token_ids[Lt]), ...]`` into flat editor inputs.

    Returns ``(flat_embeds[1,L,D], flat_position_ids[1,L], layout)`` where
    ``layout[i] = (audio_len_i, interleaved_text_len_i)``.

    ``seq_grid > 0`` (Phase 7): pad the total packed length up to a multiple of ``seq_grid`` by
    appending an *isolated filler segment* (position_ids restart at 0 -> the block-diagonal mask
    walls it off, so it cannot change any real segment's output). Its rows are NOT in ``layout``, so
    :func:`split_text_hidden` skips them. This keeps the compiled projector/editor at a few stable
    shapes -> unblocks reduce-overhead / CUDA-graph capture on the editor.
    """
    embeds_list, pos_list, layout = [], [], []
    for audio_emb, text_ids in segments:
        text_ids_i = interleave_blank(text_ids, blank_id, min_edit_len)
        text_emb = embed_tokens(text_ids_i)
        seg = torch.cat([audio_emb, text_emb], dim=0)
        embeds_list.append(seg)
        pos_list.append(torch.arange(seg.shape[0], device=seg.device))
        layout.append((int(audio_emb.shape[0]), int(text_ids_i.shape[0])))
    flat_embeds = torch.cat(embeds_list, dim=0)
    flat_position_ids = torch.cat(pos_list, dim=0)

    L = flat_embeds.shape[0]
    if seq_grid and L % seq_grid != 0:
        pad_len = seq_grid - (L % seq_grid)
        filler = flat_embeds.new_zeros(pad_len, flat_embeds.shape[-1])
        flat_embeds = torch.cat([flat_embeds, filler], dim=0)
        # position_ids restart at 0 -> isolated segment (walled off by the bidirectional mask)
        filler_pos = torch.arange(pad_len, device=flat_position_ids.device, dtype=flat_position_ids.dtype)
        flat_position_ids = torch.cat([flat_position_ids, filler_pos], dim=0)

    return flat_embeds.unsqueeze(0), flat_position_ids.unsqueeze(0), layout


def text_segment_lengths(layout, total_len: int) -> list[int]:
    """Segment split-sizes for a packed hidden/logit tensor of ``total_len`` rows.

    Appends the trailing filler chunk (Phase 7 padding) as one extra even-indexed block so a later
    ``[1::2]`` slice selects exactly the TEXT segments and ignores audio + filler.
    """
    seg_lengths = [ln for (a, t) in layout for ln in (a, t)]
    used = sum(seg_lengths)
    if total_len > used:
        seg_lengths.append(total_len - used)   # filler at an even index -> not selected by [1::2]
    return seg_lengths


def split_text_hidden(hidden: torch.Tensor, layout):
    """Split a ``[L, D]`` hidden state into the per-segment TEXT chunks (drop audio + filler)."""
    seg_lengths = text_segment_lengths(layout, hidden.shape[0])
    parts = hidden.split(seg_lengths)
    return list(parts[1::2])  # text segments are the odd slots


def text_position_mask(layout, total_len: int, device=None) -> torch.Tensor:
    """Boolean ``[total_len]`` mask, True only on the interleaved-TEXT rows (odd segments).

    Audio rows and the Phase-7 trailing filler are False. Used by the early-exit stability probe
    (llm.py) so it compares only text positions, per the Phase-5 "TEXT positions only" guarantee.
    """
    mask = torch.zeros(total_len, dtype=torch.bool, device=device)
    off = 0
    for j, ln in enumerate(text_segment_lengths(layout, total_len)):
        if j % 2 == 1:                 # text segments are at odd indices; filler is appended even
            mask[off:off + ln] = True
        off += ln
    return mask
