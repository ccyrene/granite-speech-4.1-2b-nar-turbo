"""Conformer CTC speech encoder (pure torch).

Decomposed into the smallest reusable blocks. Parameter names mirror the
checkpoint keys (``encoder.layers.{i}.{ff1,attn,conv,ff2,post_norm}`` etc.) so a
plain ``load_state_dict(strict=True)`` works.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import EncoderConfig
from . import attn_backends


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
class ConformerFeedForward(nn.Module):
    """pre_norm -> up_proj -> SiLU -> down_proj (used at half-step weight)."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(config.hidden_dim, config.hidden_dim * config.feedforward_mult)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.down_proj = nn.Linear(config.hidden_dim * config.feedforward_mult, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.up_proj(x)
        x = self.dropout(self.silu(x))
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class ConformerAttention(nn.Module):
    """Block-local attention with Shaw relative positional embeddings."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.dim_head * config.num_heads
        self.max_pos_emb = config.max_pos_emb
        self.context_size = config.context_size
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.scale = self.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(config.hidden_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1, self.dim_head)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, attention_dists: torch.Tensor,
                pad_drop: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        bsz, num_features, _ = hidden_states.shape
        cs = self.context_size

        # Branch-free pad to a multiple of context_size (pad=0 is a no-op): no data-dependent `if`
        # and no math.ceil, so torch.export traces it for a static seq (the `if remainder>0` +
        # `mask[:rem,:rem]` slice were what raised ConstraintViolation / "step sign" on export).
        pad = (cs - num_features % cs) % cs
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
        padded = num_features + pad
        num_blocks = padded // cs

        query_states = self.to_q(hidden_states)
        key_states, value_states = self.to_kv(hidden_states).chunk(2, dim=-1)

        query_states = query_states.reshape(bsz, num_blocks, cs, self.num_heads, -1).transpose(2, 3)
        key_states = key_states.reshape(bsz, num_blocks, cs, self.num_heads, -1).transpose(2, 3)
        value_states = value_states.reshape(bsz, num_blocks, cs, self.num_heads, -1).transpose(2, 3)

        # Shaw's relative positional embedding -> additive bias (already scaled).
        # Equivalent to einsum("b m h c d, c r d -> b m h c r", q, rel) without einsum:
        # the query position `c` is shared by both operands and contracted over `d`,
        # so `c` is the batch dim of a 3D matmul. We fold (b,m,h) into one dim and
        # move `c` to the front. (query_states is non-contiguous here, so the reshape
        # materializes a contiguous copy in logical order — values are unchanged.)
        # The gathered (c, r, d) table is a constant of (context_size, max_pos_emb); use the
        # per-layer cache when the loader materialized it (P2.4b) instead of re-gathering
        # ~5 MB from the embedding every layer every call. Same values, same layout.
        rel_pos_emb = getattr(self, "rel_pos_table", None)
        if rel_pos_emb is None:
            rel_pos_emb = self.rel_pos_emb(attention_dists)  # (c, r, d)
        b, m, h, c, d = query_states.shape
        r = rel_pos_emb.shape[1]
        n = b * m * h
        # (b,m,h,c,d) -> (N,c,d) -> (c,N,d) ; (c,r,d) -> (c,d,r) ; matmul -> (c,N,r)
        q3d = query_states.reshape(n, c, d).transpose(0, 1)
        pos = torch.matmul(q3d, rel_pos_emb.transpose(1, 2))
        # (c,N,r) -> (N,c,r) -> (b,m,h,c,r)
        pos_attn = pos.transpose(0, 1).reshape(b, m, h, c, r) * self.scale

        # Branch-free padding mask (replaces `if remainder>0: mask[:rem,:rem]=0` on the last block):
        # within a block, attend only between real (non-pad) positions. Full blocks are all-real ->
        # keep everything -> no masking, exactly matching the original (which only touched the last block).
        # ``pad_drop`` (P2.4a) is the ~keep mask hoisted out of the 16-layer loop by the encoder
        # (it depends only on (num_features, cs)); rebuilt here when called standalone.
        if pad_drop is None:
            valid = (torch.arange(padded, device=hidden_states.device) < num_features).reshape(num_blocks, cs)
            keep = valid.unsqueeze(-1) & valid.unsqueeze(-2)        # (m, c, c)
            pad_drop = ~keep[None, :, None, :, :]
        pos_attn = pos_attn.masked_fill(pad_drop, -torch.finfo(pos_attn.dtype).max)

        if attn_backends.encoder_fused_enabled() and hidden_states.is_cuda:
            # Opt-in fused path (GRANITE_ENC_ATTN=fused): fold (b, blocks) into the batch dim
            # (5-D -> 4-D views; flatten(0,1) stays a view after the transpose above) so SDPA
            # may pick a fused backend (cuDNN / mem-efficient) with the additive bias. 5-D
            # inputs are what force the MATH backend below. Fully-padded rows keep the finite
            # -finfo.max bias -> softmax stays NaN-free; they are sliced off after to_out.
            out = F.scaled_dot_product_attention(
                query_states.flatten(0, 1), key_states.flatten(0, 1),
                value_states.flatten(0, 1), attn_mask=pos_attn.flatten(0, 1), scale=self.scale,
            ).reshape(b, m, h, c, -1)
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                out = F.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=pos_attn, scale=self.scale
                )
        out = out.transpose(2, 3).reshape(bsz, padded, -1)
        out = self.to_out(out[:, :num_features, :])
        return self.dropout(out)


class DepthWiseConv1d(nn.Module):
    """Padded depthwise 1D convolution (key: ``.conv.weight``).

    P2.4d: when the padding is symmetric (odd kernel, e.g. k=15 -> (7,7)) the zero-pad is
    folded into the conv itself, deleting one full (B, C, T) copy per call (x16 layers).
    Bitwise-identical to F.pad+conv (implicit zero padding is the same convolution; verified
    on cudnn fp32/bf16). Asymmetric kernels keep the explicit F.pad."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)
        self._symmetric = (pad_offset == 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in, bias=False,
                              padding=(pad if self._symmetric else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._symmetric:
            return self.conv(x)
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """LN -> pointwise up -> GLU -> depthwise -> BN -> SiLU -> pointwise down."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depth_conv = DepthWiseConv1d(inner_dim, inner_dim, kernel_size=config.conv_kernel_size)
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)
        self.dropout = nn.Dropout(config.dropout)
        # set by conv_kernel.enable_dwconv_silu (post BN-fold): False | "dwconv_silu" | "dwconv_silu_glu"
        self._fused_conv = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.up_conv(x.permute(0, 2, 1))
        if self._fused_conv == "dwconv_silu_glu":
            x = torch.ops.granite.dwconv_silu_glu(x, self.depth_conv.conv.weight, self.depth_conv.conv.bias)
        else:
            x = self.glu(x)
            if self._fused_conv:
                x = torch.ops.granite.dwconv_silu(x, self.depth_conv.conv.weight, self.depth_conv.conv.bias)
            else:
                x = self.depth_conv(x)
                x = self.silu(self.batch_norm(x))
        x = self.down_conv(x).permute(0, 2, 1)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """FF(½) -> Attn -> Conv -> FF(½) -> post LN, all residual."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.ff1 = ConformerFeedForward(config)
        self.attn = ConformerAttention(config)
        self.conv = ConformerConvModule(config)
        self.ff2 = ConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, attention_dists: torch.Tensor,
                pad_drop: torch.Tensor | None = None) -> torch.Tensor:
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x, attention_dists=attention_dists, pad_drop=pad_drop) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.post_norm(x)
        return x


# --------------------------------------------------------------------------- #
# Posterior-weighted pooling (BPE head front-end)
# --------------------------------------------------------------------------- #
def posterior_weighted_pool(hidden: torch.Tensor, importance: torch.Tensor, window_size: int = 4) -> torch.Tensor:
    batch_size, seq_len, hidden_dim = hidden.shape
    pad_len = (window_size - seq_len % window_size) % window_size
    if pad_len > 0:
        hidden = F.pad(hidden, (0, 0, 0, pad_len))
        importance = F.pad(importance, (0, pad_len))
    num_windows = hidden.shape[1] // window_size
    hidden = hidden.view(batch_size, num_windows, window_size, hidden_dim)
    importance = importance.view(batch_size, num_windows, window_size)
    weights = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
    pooled = (hidden * weights.unsqueeze(-1)).sum(dim=2)
    return pooled


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
@dataclass
class EncoderOutput:
    logits: torch.Tensor | None = None          # flat BPE CTC logits
    last_hidden_state: torch.Tensor | None = None
    all_hidden_states: tuple[torch.Tensor, ...] | None = None


class CTCEncoder(nn.Module):
    """16-layer Conformer with dual CTC head, self-conditioning, BPE pooling head."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(config.num_layers)])
        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)
        self.out_bpe = None
        if config.bpe_output_dim is not None:
            self.out_bpe = nn.Linear(config.hidden_dim, config.bpe_output_dim, bias=True)
        self.dropout = nn.Dropout(config.pred_dropout)

        # Block-local attention distances are constant for (context_size, max_pos_emb); cache them as a
        # non-persistent buffer (Tier-1 #3) so forward() doesn't rebuild arange/clamp every call.
        # int64 + non-persistent: excluded from state_dict (assign=True load untouched), moved by .to(device).
        cs = config.context_size
        _seq = torch.arange(cs)
        self.register_buffer(
            "attention_dists",
            torch.clamp(_seq.view(-1, 1) - _seq.view(1, -1), -cs, cs) + config.max_pos_emb,
            persistent=False,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.input_linear.weight.dtype

    def cache_rel_pos_tables(self):
        """P2.4b: materialize each layer's gathered Shaw rel-pos table (a constant of
        (context_size, max_pos_emb)) as a non-persistent buffer, so forward() stops
        re-gathering ~5 MB per layer per call. Call after weights are loaded/moved."""
        dists = self.attention_dists
        for layer in self.layers:
            attn = layer.attn
            with torch.no_grad():
                table = attn.rel_pos_emb(dists.to(attn.rel_pos_emb.weight.device))
            attn.register_buffer("rel_pos_table", table.contiguous(), persistent=False)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        hidden_state_indices: list[int] | None = None,
    ) -> EncoderOutput:
        if attention_mask is None:
            attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.bool, device=input_features.device)

        hidden_states = self.input_linear(input_features.to(self.dtype))
        blank_probs = None
        attention_dists = self.attention_dists

        # P2.4a: the block-local pad mask depends only on (seq_len, context_size) — build it ONCE
        # here instead of once per layer. Bit-exact: same mask every layer.
        cs = self.config.context_size
        T = hidden_states.shape[1]
        pad = (cs - T % cs) % cs
        valid = (torch.arange(T + pad, device=hidden_states.device) < T).reshape((T + pad) // cs, cs)
        keep = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        pad_drop = ~keep[None, :, None, :, :]                       # (1, m, 1, c, c)

        # Hidden-state collection. Default (hidden_state_indices=None): full tuple
        # (index 0 = embeddings, i = output of layer i) — preserves the verification API.
        # If indices are given (Tier-1 #2), keep ONLY those layers, returned as a dict keyed by the
        # requested index, so the 13 unused intermediate activations are freed during the loop.
        all_hidden_states = None
        keep = None       # {normalized_idx: requested_idx}
        if output_hidden_states and hidden_state_indices is not None:
            total = self.config.num_layers + 1
            keep = {(i if i >= 0 else total + i): i for i in hidden_state_indices}
            selected = {req: hidden_states for norm, req in keep.items() if norm == 0}
        elif output_hidden_states:
            all_hidden_states = (hidden_states,)

        for layer_idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=attention_dists, pad_drop=pad_drop)

            if layer_idx == self.config.self_conditioning_layer:
                mid_logits = self.out(self.dropout(hidden_states))
                mid_probs = torch.softmax(mid_logits.float(), dim=-1)
                blank_probs = mid_probs[:, :, 0]
                hidden_states = hidden_states + self.out_mid(mid_probs.to(hidden_states.dtype))

            if output_hidden_states:
                if keep is None:
                    all_hidden_states += (hidden_states,)
                elif layer_idx in keep:
                    selected[keep[layer_idx]] = hidden_states

        if keep is not None:
            all_hidden_states = selected

        hidden_states = self.dropout(hidden_states)

        logits = None
        if self.out_bpe is not None and blank_probs is not None:
            pool_window = self.config.bpe_pooling_window
            importance = 1.0 - blank_probs
            pooled = posterior_weighted_pool(hidden_states.float(), importance, window_size=pool_window).to(
                hidden_states.dtype
            )
            if attn_backends.encoder_dense_bpe_enabled():
                # P3.1 (GRANITE_ENC_DENSE_BPE=1): run the BPE head densely on the padded pooled
                # tensor — no lengths/`.tolist()` host sync inside the compiled graph, so the
                # encoder compiles as ONE graph. Pad rows are dropped by the caller using host
                # lengths. The GEMM M-dim changes vs the packed path -> transcript-safe class
                # (argmax-gated), not bit-exact.
                logits = self.out_bpe(pooled)                       # (B, P, V) dense
            else:
                encoder_lengths = attention_mask.sum(dim=1)
                lengths = -(encoder_lengths // -pool_window)
                lengths_list = lengths.tolist()
                logits = self.out_bpe(torch.cat([pooled[i, :length] for i, length in enumerate(lengths_list)]))

        return EncoderOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states,
        )
