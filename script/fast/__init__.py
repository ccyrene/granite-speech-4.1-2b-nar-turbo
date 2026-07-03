"""Optimized adaptive CTC-first inference for Granite Speech 4.1 2B NAR (pure torch.compile, no TensorRT)."""
from .fast_asr import FastGraniteASR

__all__ = ["FastGraniteASR"]
