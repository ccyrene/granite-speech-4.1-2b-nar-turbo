"""Bidirectional Granite LLM editor (pure torch).

A 40-layer Granite decoder stack with the causal mask removed (non-autoregressive
/ bidirectional). Granite specifics: embedding/residual/attention multipliers,
``logits_scaling``, GQA, RoPE, RMSNorm, tied input/output embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import TextConfig
from .masking import build_bidirectional_mask
from . import attn_backends, kernels

ACT = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class RMSNorm(nn.Module):
    """T5-style RMSNorm computed in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Opt-in fused kernel (Phase 6); default OFF -> the original bit-exact path below.
        if kernels.enabled() and hidden_states.is_cuda:
            return kernels.rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig, device=None):
        super().__init__()
        dim = config.head_dim
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    """Granite GQA attention, bidirectional (is_causal=False)."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = config.attention_multiplier

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if attn_backends.is_block_mask(attention_mask):
            # Opt-in flex path (GRANITE_EDITOR_ATTN=flex): block-diagonal BlockMask skips
            # cross-segment tiles -> O(sum L_i^2) instead of dense O(L_total^2); GQA native.
            attn_output = attn_backends.flex_attention_gqa(q, k, v, attention_mask, self.scaling)
        else:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, scale=self.scaling, is_causal=False
            )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT[config.hidden_act]
        self._is_silu = config.hidden_act == "silu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Opt-in fused silu(gate)*up kernel (Phase 6); default OFF -> original math (bit-identical).
        if self._is_silu and kernels.enabled() and x.is_cuda:
            act = kernels.silu_mul(gate, up)
        else:
            act = self.act_fn(gate) * up
        return self.down_proj(act)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_multiplier = config.residual_multiplier

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


@dataclass
class LMOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class GraniteModel(nn.Module):
    """Embeddings + decoder stack + final norm."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.embedding_multiplier = config.embedding_multiplier

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                attention_mask=None, early_exit=None, text_position_mask=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # NOTE: applies to externally-supplied inputs_embeds too (matches reference).
        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        # NOTE: masking here is specific to this model's *pad-free* packing, where
        # position_ids reset to 0 at each new packed sample. A 0 means "new sample",
        # NOT padding. This is not a general Granite batch mask (a conventional
        # right-padded batch with trailing zero positions would be mis-segmented).
        if attention_mask is None and attn_backends.editor_flex_enabled() and inputs_embeds.is_cuda:
            attention_mask = attn_backends.build_editor_block_mask(position_ids)
        if attention_mask is None:   # flex off / unavailable / single segment -> original path
            attention_mask = build_bidirectional_mask(position_ids)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Early-exit controller (Phase 5). Default: early_exit=None -> tracker None -> the loop is
        # byte-identical to the baseline. NOTE: data-dependent break -> not torch.compile / CUDA-graph
        # compatible; use only on the eager adaptive path.
        tracker = None
        if early_exit is not None and getattr(early_exit, "enabled", False):
            from .adaptive.early_exit import StabilityTracker
            tracker = StabilityTracker(early_exit)
        n_layers = len(self.layers)

        for li, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
            if (tracker is not None and (li + 1) >= early_exit.min_layers
                    and (li + 1) % early_exit.check_every == 0 and (li + 1) < n_layers):
                h = self.norm(hidden_states).squeeze(0)                 # [L, D]
                logits = F.linear(h, self.embed_tokens.weight)          # tied head (pre-scaling)
                if text_position_mask is not None:
                    logits = logits[text_position_mask.reshape(-1)]     # TEXT positions only (Phase 5)
                if logits.numel():
                    ids = logits.argmax(dim=-1)                         # argmax invariant to /scaling
                    # confidence MUST use the model's true (scaled) distribution, else it is inflated
                    probs = torch.softmax((logits / self.config.logits_scaling).float(), dim=-1)
                    conf = probs.max(dim=-1).values
                    if tracker.update(ids, float(conf.mean())):
                        break

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GraniteLM(nn.Module):
    """LM head tied to the input embeddings (no separate ``lm_head.weight``)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = GraniteModel(config)
        self.logits_scaling = config.logits_scaling

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                attention_mask=None, early_exit=None, text_position_mask=None) -> LMOutput:
        hidden_states = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                   position_ids=position_ids, attention_mask=attention_mask, early_exit=early_exit,
                                   text_position_mask=text_position_mask)
        logits = F.linear(hidden_states, self.model.embed_tokens.weight)  # tied
        logits = logits / self.logits_scaling
        return LMOutput(logits=logits, last_hidden_state=hidden_states)
