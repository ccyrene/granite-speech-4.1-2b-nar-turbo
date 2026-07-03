"""Windowed Q-Former projector (pure torch).

Maps the concatenated multi-layer encoder features (4 layers x 1024 = 4096) into
LLM embedding space, downsampling 5x within 15-frame windows via cross-attention
from 3 learned queries.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ProjectorConfig


class QFormerCrossAttention(nn.Module):
    """Cross-attention: queries attend to encoder window features."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        encoder_len = encoder_hidden_states.shape[1]

        q = self.q_proj(hidden_states).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(batch_size, encoder_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(batch_size, encoder_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # .reshape (== .contiguous().view) — export/TRT-friendly on the transposed tensor
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.hidden_size)
        return self.o_proj(attn_output)


class QFormerMLP(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        mlp_hidden_size = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, mlp_hidden_size, bias=config.mlp_bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_hidden_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class QFormerLayer(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.cross_attention = QFormerCrossAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.mlp = QFormerMLP(config)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.cross_attention(self.attn_norm(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class QFormer(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layers = nn.ModuleList([QFormerLayer(config) for _ in range(config.num_layers)])

    def forward(self, query_embeds: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = query_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class Projector(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.encoder_dim, eps=config.layernorm_eps) for _ in range(config.num_encoder_layers)]
        )
        self.layer_projector = nn.Linear(config.encoder_dim * config.num_encoder_layers, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.projector_act = nn.GELU()
        self.qformer = QFormer(config)

        query_length = config.block_size // config.downsample_rate
        embed_std = config.hidden_size**-0.5
        self.query = nn.Parameter(torch.randn(1, query_length, config.hidden_size) * embed_std)
        self.window_positions = nn.Parameter(torch.randn(1, config.block_size, config.hidden_size) * embed_std)
        self.out_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        self.out_linear = nn.Linear(config.hidden_size, config.llm_dim)

    def forward(self, hidden_states) -> torch.Tensor:
        if isinstance(hidden_states, (list, tuple)):
            # P2.4f: accept the per-layer encoder states directly and normalize each before the
            # single post-norm cat — the caller's big (B, T, layers*dim) pre-cat copy disappears.
            # Bit-exact: each LayerNorm sees the same rows (contiguous either way).
            batch_size, seq_len = hidden_states[0].shape[0], hidden_states[0].shape[1]
            normalized_layers = [ln(h) for ln, h in zip(self.layer_norms, hidden_states)]
            hidden_states = torch.cat(normalized_layers, dim=-1)
        else:
            batch_size, seq_len, dim = hidden_states.size()

            hidden_states = hidden_states.view(
                batch_size, seq_len, self.config.num_encoder_layers, self.config.encoder_dim
            )
            normalized_layers = [ln(hidden_states[:, :, i]) for i, ln in enumerate(self.layer_norms)]
            hidden_states = torch.cat(normalized_layers, dim=-1)

        hidden_states = self.projector_act(self.layer_projector(hidden_states))

        block_size = self.config.block_size
        # Branch-free zero-pad to a multiple of block_size (no-op when already
        # aligned). Equivalent to `if rest>0: pad`, but with no data-dependent
        # Python branch -> exports cleanly to ONNX/TensorRT with a dynamic seq dim.
        pad_amt = (block_size - seq_len % block_size) % block_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_amt), "constant", 0)
        nblocks = (seq_len + pad_amt) // block_size

        hidden_states = hidden_states.reshape(batch_size * nblocks, block_size, self.config.hidden_size)
        query_length = self.query.shape[1]
        mean_pool = hidden_states.reshape(
            batch_size * nblocks, query_length, self.config.downsample_rate, self.config.hidden_size
        ).mean(dim=-2)

        hidden_states = self.qformer(
            query_embeds=self.dropout(self.query + mean_pool),
            encoder_hidden_states=self.dropout(hidden_states + self.window_positions),
        )

        hidden_states = hidden_states.reshape(batch_size, nblocks * query_length, -1)
        hidden_states = self.dropout(self.out_norm(hidden_states))
        return self.out_linear(hidden_states)
