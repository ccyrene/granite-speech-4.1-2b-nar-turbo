"""Length-aware batch packing (OPTIMIZATION_PLAN Phase 8).

Shape bucketing pads every sample to the bucket length; if a bucket mixes very short and very long
samples, most of the batch is padding. Sorting by *actual* length and grouping similar lengths
together shrinks the padding each batch pays for. Pure book-keeping over lengths — no tensors, so
it is trivially unit-testable and used by ``transcribe_long`` / the bench harness to order windows.
"""
from __future__ import annotations


def padding_ratio(lengths: list[int], bucket_len: int | None = None) -> float:
    """padded_frames / actual_frames for one batch.

    If ``bucket_len`` is given, every sample is padded to it; otherwise to the batch max
    (right-padding a ragged batch). Ratio 1.0 means no wasted compute.
    """
    if not lengths:
        return 1.0
    actual = sum(lengths)
    if actual == 0:
        return 1.0
    pad_to = bucket_len if bucket_len is not None else max(lengths)
    padded = pad_to * len(lengths)
    return padded / actual


def length_aware_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Group sample indices into batches of near-equal length.

    Returns a list of batches, each a list of ORIGINAL indices (so the caller can reorder outputs
    back). Samples are sorted by length then chunked, so each batch mixes only similar lengths and
    pays minimal padding. Order-preserving reconstruction is the caller's job (keep the index map).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def avg_padding_ratio(lengths: list[int], batches: list[list[int]]) -> float:
    """Mean per-batch padding ratio (each batch padded to its own max) — a scalar to track."""
    if not batches:
        return 1.0
    ratios = [padding_ratio([lengths[i] for i in b]) for b in batches if b]
    return sum(ratios) / len(ratios) if ratios else 1.0
