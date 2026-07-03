"""Long-form audio handling for FastGraniteASR: split >max_s audio into overlapping windows, transcribe
each, then stitch the per-window transcripts back exactly by de-duplicating the overlap region.

Why overlap + merge (not hard 30s cuts): a hard boundary cuts a word in half -> both neighbouring windows
mis-transcribe it. With an N-second overlap the boundary word is seen whole by at least one window; the
merge finds the longest matching run of words inside the overlap and joins there, dropping the duplicate.

Why not zero-pad the waveform to exactly max_s: that makes the model see trailing silence as real audio
(mask would mark it valid -> possible hallucination). We pass the real-length chunk to the feature
extractor, which builds the attention_mask from the true length -> the model masks padding (bit-exact).
The FRAME_GRID bucketing then keeps the compiled encoder shape stable.
"""
from __future__ import annotations

import difflib

import torch


def chunk_waveform(wav: torch.Tensor, sr: int = 16000, max_s: float = 30.0, overlap_s: float = 5.0):
    """Split a 1-D waveform into <=max_s windows with overlap_s overlap. <=max_s -> returned as-is."""
    wav = wav.reshape(-1)
    n = wav.shape[0]
    win = int(max_s * sr)
    if n <= win:
        return [wav]
    stride = int((max_s - overlap_s) * sr)
    chunks, start = [], 0
    while start < n:
        chunks.append(wav[start:start + win])
        if start + win >= n:
            break
        start += stride
    return chunks


def merge_words(window_words: list[list[str]], overlap_window: int = 40, min_match: int = 2) -> list[str]:
    """Stitch consecutive windows' word-lists, removing the duplicated overlap via the longest matching
    run found inside the trailing/leading `overlap_window` words. Falls back to plain concat if no
    reliable (>=min_match-word) overlap is found."""
    if not window_words:
        return []
    merged = list(window_words[0])
    for nxt in window_words[1:]:
        if not nxt:
            continue
        if not merged:
            merged = list(nxt)
            continue
        W = min(overlap_window, len(merged), len(nxt))
        a_tail, b_head = merged[-W:], nxt[:W]
        m = difflib.SequenceMatcher(None, a_tail, b_head, autojunk=False).find_longest_match(0, len(a_tail), 0, len(b_head))
        if m.size >= min_match:
            a_cut = len(merged) - W + m.a + m.size   # keep merged up to end of the matched run
            b_cut = m.b + m.size                     # take next from end of the matched run
            merged = merged[:a_cut] + nxt[b_cut:]
        else:
            merged = merged + nxt
    return merged
