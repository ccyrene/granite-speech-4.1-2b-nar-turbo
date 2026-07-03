"""Opt-in attention backends (default **OFF** -> the original bit-exact paths run unchanged).

Two independent levers, read from the environment at *call time* (so an A/B harness can flip
them between runs without reloading 4.3 GB of weights):

  GRANITE_EDITOR_ATTN=flex
      Packed LLM editor: FlexAttention + block-diagonal BlockMask. The packed ``[1, L_total]``
      batch currently runs dense bool-masked SDPA -> O(L_total^2) compute plus an O(L_total^2)
      bool mask tensor, so per-sample editor cost grows ~linearly with batch size (the measured
      H100 b16->b128 RTFx collapse). A block-diagonal
      BlockMask skips cross-segment tiles entirely -> O(sum L_i^2), batch-size-independent,
      and GQA is handled natively (``enable_gqa=True``, no ``repeat_kv`` materialization).

  GRANITE_ENC_ATTN=fused
      Conformer block-local attention: fold ``(b, blocks)`` into the batch dim (5-D -> 4-D)
      so SDPA may pick a fused backend (cuDNN / mem-efficient) with the additive Shaw bias,
      instead of the MATH backend that 5-D inputs force today (conformer.py). Padded rows
      keep the finite ``-finfo.max`` bias (not -inf) -> no NaN rows; they are sliced off by
      the caller exactly as before.

Both levers change *kernels*, not math: outputs are argmax/transcript-stable with small float
drift (fp32-accumulated fused softmax), NOT bit-identical -> same validation bar as the
``compile-*`` levers (max|dlogit| + preds match). Parity + speed A/B: ``script/bench_attn_ab.py``.
"""
from __future__ import annotations

import os

import torch

from .masking import packed_segment_ids

_FLEX_COMPILED = None
_FLEX_RAW = None
_FLEX_KOPTS = None


def _flex_kernel_options() -> dict | None:
    """flex_attention's default 128x128 tiles need >100 KB shared memory — fine on A100
    (164 KB) / H100 (228 KB), over the limit on consumer sm_86/sm_89 (99 KB). Drop to
    64x64 tiles there so the same lever runs on the dev card."""
    global _FLEX_KOPTS
    if _FLEX_KOPTS is None:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (9, 0)
        big_smem = cap == (8, 0) or cap >= (9, 0)
        _FLEX_KOPTS = {} if big_smem else {"BLOCK_M": 64, "BLOCK_N": 64}
    return _FLEX_KOPTS or None


def editor_flex_enabled() -> bool:
    return os.environ.get("GRANITE_EDITOR_ATTN", "").strip().lower() == "flex"


def encoder_fused_enabled() -> bool:
    return os.environ.get("GRANITE_ENC_ATTN", "").strip().lower() == "fused"


def encoder_dense_bpe_enabled() -> bool:
    """P3.1 lever (GRANITE_ENC_DENSE_BPE=1): dense BPE CTC head — the encoder emits (B, P, V)
    padded logits with no host sync in-graph; callers slice by host lengths. Read at call time
    inside the compiled encoder -> baked as a dynamo guard: set BEFORE warmup, never flip."""
    return os.environ.get("GRANITE_ENC_DENSE_BPE", "").strip() == "1"


def projector_slice_enabled() -> bool:
    """P2.3 sub-lever (GRANITE_PROJ_SLICE=1): run the projector only on rows routed to the
    editor (ulp-class: GEMM M-dim changes) — gate separately from the bit-exact pack."""
    return os.environ.get("GRANITE_PROJ_SLICE", "").strip() == "1"


def conv_kernel() -> str:
    """Exp1 lever (GRANITE_CONV_KERNEL=dwconv_silu): fold BN and run the Conformer conv
    module's depthwise conv+bias+SiLU as one custom op (see conv_kernel.py). WER-gated,
    not bit-exact (the BN fold rounds differently)."""
    return os.environ.get("GRANITE_CONV_KERNEL", "").strip().lower()


def texthead_argmax_enabled() -> bool:
    """Exp2 lever (GRANITE_TEXTHEAD_ARGMAX=1): editor text head as chunked GEMM + running
    argmax — full-vocab logits never materialize (cuda_kernels/texthead_argmax.py). Only
    affects the adaptive texthead path; ulp-class (chunked cuBLAS kernel selection)."""
    return os.environ.get("GRANITE_TEXTHEAD_ARGMAX", "").strip() == "1"


def is_block_mask(obj) -> bool:
    """True iff ``obj`` is a FlexAttention BlockMask (cheap: no flex import on the hot path
    unless the flex lever actually produced one)."""
    return obj is not None and type(obj).__name__ == "BlockMask"


def _segment_ids_from_lengths(segment_lengths: list[int], device) -> torch.Tensor:
    ids = []
    for i, ln in enumerate(segment_lengths):
        ids.extend([i] * int(ln))
    return torch.tensor(ids, dtype=torch.int64, device=device)


@torch.compiler.disable(recursive=False)
def build_editor_block_mask_from_lengths(segment_lengths: list[int], device, total_len: int | None = None):
    """Build a FlexAttention BlockMask from host segment lengths.

    This avoids deriving segments from ``position_ids`` on device and avoids ``create_block_mask``'s
    per-forward dense mask conversion. ``mask_mod`` still enforces exact token boundaries for
    segments that do not align to sparse block boundaries.
    """
    if (
        not editor_flex_enabled()
        or os.environ.get("GRANITE_EDITOR_HOST_BLOCKMASK", "").strip() != "1"
        or not torch.cuda.is_available()
    ):
        return None
    lengths = [int(x) for x in segment_lengths if int(x) > 0]
    if len(lengths) <= 1:
        return None
    L = int(total_len) if total_len is not None else sum(lengths)
    if L <= 0:
        return None
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except Exception:
        return None

    block_size = 128
    q_blocks = (L + block_size - 1) // block_size
    bounds = []
    off = 0
    for ln in lengths:
        bounds.append((off, off + ln))
        off += ln
    if off < L:
        bounds.append((off, L))

    rows = []
    max_cols = 0
    for qb in range(q_blocks):
        q0, q1 = qb * block_size, min((qb + 1) * block_size, L)
        cols = set()
        for s0, s1 in bounds:
            if s0 < q1 and s1 > q0:
                cols.update(range(s0 // block_size, (s1 - 1) // block_size + 1))
        cols = sorted(cols) or [0]
        rows.append(cols)
        max_cols = max(max_cols, len(cols))

    kv_num_blocks = torch.tensor([[[len(cols) for cols in rows]]], dtype=torch.int32, device=device)
    kv_indices_host = [[cols + [0] * (max_cols - len(cols)) for cols in rows]]
    kv_indices = torch.tensor([kv_indices_host], dtype=torch.int32, device=device)

    seg = _segment_ids_from_lengths([s1 - s0 for s0, s1 in bounds], device)

    def mask_mod(b, h, q_idx, kv_idx):
        in_bounds = (q_idx < L) & (kv_idx < L)
        q_safe = torch.where(q_idx < L, q_idx, torch.zeros_like(q_idx))
        kv_safe = torch.where(kv_idx < L, kv_idx, torch.zeros_like(kv_idx))
        return in_bounds & (seg[q_safe] == seg[kv_safe])

    try:
        return BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, BLOCK_SIZE=block_size, mask_mod=mask_mod, seq_lengths=(L, L)
        )
    except Exception:
        # Version/API fallback: keep the old create_block_mask route available.
        try:
            from torch.nn.attention.flex_attention import create_block_mask
            return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                     device=device, BLOCK_SIZE=block_size, _compile=True)
        except Exception:
            return None


def _flex_fns():
    """(raw, compiled) flex_attention. Raw is used inside an outer torch.compile graph
    (inductor lowers it natively); the compiled wrapper is used from eager (eager flex is a
    slow reference implementation)."""
    global _FLEX_COMPILED, _FLEX_RAW
    if _FLEX_RAW is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX_RAW = flex_attention
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_RAW, _FLEX_COMPILED


@torch.compiler.disable(recursive=False)
def build_editor_block_mask(position_ids: torch.Tensor):
    """BlockMask for the pad-free packed editor batch, or ``None`` to use the original path.

    ``None`` when: flex unavailable, not the packed ``[1, L]`` layout, or a single segment
    (where the original path already runs unmasked full attention -> nothing to skip).
    Built once per forward and shared by all 40 layers. ``recursive=False``: this frame is
    kept out of an outer compile graph (data-dependent mask contents), but the internal
    ``_compile=True`` block-mask builder still compiles.
    """
    if position_ids.shape[0] != 1:
        return None
    seg = packed_segment_ids(position_ids)  # (1, L)
    if seg.max() <= 1:
        return None
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except Exception:
        return None
    s = seg[0].contiguous()
    L = s.shape[0]

    def mask_mod(b, h, q_idx, kv_idx):
        return s[q_idx] == s[kv_idx]

    try:
        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L,
                                 device=s.device, _compile=True)
    except Exception:
        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=s.device)


def flex_attention_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       block_mask, scale: float) -> torch.Tensor:
    """flex_attention with native GQA (k/v stay at num_kv_heads; no repeat_kv copies)."""
    raw, compiled = _flex_fns()
    fn = raw if torch.compiler.is_compiling() else compiled
    return fn(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=True,
              kernel_options=_flex_kernel_options())
