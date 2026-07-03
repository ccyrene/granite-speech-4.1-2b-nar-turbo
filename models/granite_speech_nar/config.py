"""Plain-dataclass configuration for Granite Speech NAR.

Mirrors the fields of the original transformers ``PreTrainedConfig`` subclasses but
carries zero dependency on ``transformers``. Built directly from the model's
``config.json`` so values stay faithful to the checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EncoderConfig:
    """Conformer CTC encoder."""

    input_dim: int = 160
    num_layers: int = 16
    hidden_dim: int = 1024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int | None = None
    output_dim: int = 348
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float = 0.1
    pred_dropout: float = 0.25
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2
    self_conditioning_layer: int | None = None
    bpe_output_dim: int | None = None
    bpe_pooling_window: int = 4
    blank_token_id: int | None = None

    def __post_init__(self):
        if self.dim_head is None:
            self.dim_head = self.hidden_dim // self.num_heads
        if self.self_conditioning_layer is None:
            self.self_conditioning_layer = self.num_layers // 2


@dataclass
class ProjectorConfig:
    """Windowed Q-Former projector."""

    encoder_dim: int = 1024
    llm_dim: int = 2048
    downsample_rate: int = 5
    num_encoder_layers: int = 4
    hidden_size: int = 2048
    num_heads: int = 32
    num_layers: int = 2
    dropout_prob: float = 0.1
    block_size: int = 15
    mlp_ratio: int = 2
    layernorm_eps: float = 1e-6
    attn_bias: bool = True
    mlp_bias: bool = True


@dataclass
class TextConfig:
    """Bidirectional Granite LLM."""

    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 4096
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    attention_multiplier: float = 0.0078125
    embedding_multiplier: float = 12.0
    residual_multiplier: float = 0.22
    logits_scaling: float = 8.0
    rope_theta: float = 10000.0
    pad_token_id: int = 100256
    eos_token_id: int = 100257
    bos_token_id: int = 100257
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class GraniteSpeechNarConfig:
    """Top-level ASR config tying the three components together."""

    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    projector_config: ProjectorConfig = field(default_factory=ProjectorConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    encoder_layer_indices: list[int] = field(default_factory=lambda: [4, 8, 12, -1])
    scale_projected_embeddings: bool = True
    blank_token_id: int = 100257
    min_edit_sequence_length: int = 8

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GraniteSpeechNarConfig":
        with open(path) as f:
            cfg = json.load(f)

        def filt(dc, d):
            names = {f.name for f in dc.__dataclass_fields__.values()}
            return {k: v for k, v in d.items() if k in names}

        enc = cfg["encoder_config"]
        tc = cfg["text_config"]
        text_kwargs = filt(TextConfig, tc)
        # rope_parameters -> rope_theta
        rp = tc.get("rope_parameters", {})
        if "rope_theta" in rp:
            text_kwargs["rope_theta"] = rp["rope_theta"]

        out = cls(
            encoder_config=EncoderConfig(**filt(EncoderConfig, enc)),
            projector_config=ProjectorConfig(**filt(ProjectorConfig, cfg["projector_config"])),
            text_config=TextConfig(**text_kwargs),
            encoder_layer_indices=list(cfg.get("encoder_layer_indices", [4, 8, 12, -1])),
            scale_projected_embeddings=cfg.get("scale_projected_embeddings", True),
            blank_token_id=cfg.get("blank_token_id", 100257),
            min_edit_sequence_length=cfg.get("min_edit_sequence_length", 8),
        )
        # propagate blank id into encoder (matches reference __post_init__)
        out.encoder_config.blank_token_id = out.blank_token_id
        return out
