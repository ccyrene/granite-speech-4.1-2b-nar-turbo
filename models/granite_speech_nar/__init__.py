"""Pure-torch reimplementation of IBM Granite Speech 4.1 2B NAR.

No ``transformers`` dependency. Components:
  - config:     plain-dataclass configuration loaded from config.json
  - features:   plain-torch log-mel feature extractor (precomputed mel_filters.pt)
  - conformer:  16-layer Conformer CTC encoder (dual head, self-conditioning)
  - projector:  windowed Q-Former audio->LLM projector
  - llm:        40-layer bidirectional Granite LLM editor (tied embeddings)
  - asr:        GraniteSpeechNarForASR end-to-end (transcribe)
  - loader:     safetensors -> model
  - tokenizer:  token-id -> text via the `tokenizers` library
"""
from .config import (
    GraniteSpeechNarConfig,
    EncoderConfig,
    ProjectorConfig,
    TextConfig,
)
from .features import MelFeatureExtractor
from .conformer import CTCEncoder
from .projector import Projector
from .llm import GraniteLM, GraniteModel
from .asr import GraniteSpeechNarForASR, ASROutput
from .loader import load_model, build_model, load_state_dict_from_safetensors
from .tokenizer import SpeechTokenizer

__all__ = [
    "GraniteSpeechNarConfig",
    "EncoderConfig",
    "ProjectorConfig",
    "TextConfig",
    "MelFeatureExtractor",
    "CTCEncoder",
    "Projector",
    "GraniteLM",
    "GraniteModel",
    "GraniteSpeechNarForASR",
    "ASROutput",
    "load_model",
    "build_model",
    "load_state_dict_from_safetensors",
    "SpeechTokenizer",
]
