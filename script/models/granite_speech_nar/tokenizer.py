"""Token id -> text decoding via the standalone ``tokenizers`` library.

This is the HuggingFace *tokenizers* Rust library, not ``transformers``; it loads
``tokenizer.json`` directly.
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer


class SpeechTokenizer:
    def __init__(self, tokenizer_json_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [int(i) for i in ids]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, ids_list, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_list]
