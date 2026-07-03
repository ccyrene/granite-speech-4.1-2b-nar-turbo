"""Shared serving engine — the piece both backends (Ray Serve / Triton) wrap.

Owns everything the benchmarks proved matters at serving time:
  * duration-sorted execution chunks of <=EXEC_BATCH (the b128/exec48 lesson: length-bucketed
    chunks beat one monolithic padded batch, and avoid the b>=64 max-autotune failure class)
  * FRAME_GRID shape bucketing (inside FastGraniteASR) so torch.compile shapes stay bounded
  * warmup over the bucket grid so replicas come up hot (persist TORCHINDUCTOR_CACHE_DIR
    across restarts to skip the 5-10 min cold autotune)
  * long-form routing: clips > 30 s go through chunk+merge (transcribe_long)

Contract: inputs are mono float32 waveforms at 16 kHz (torch tensors or numpy arrays).
Ingress layers are responsible for decode/resample (see decode_wav_bytes below).
"""
from __future__ import annotations

import io
import os
import sys
import threading

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MAX_CLIP_S = 30.0
SR = 16000
# batch-dim bucketing: pad execution chunks up to the next grid size so torch.compile sees a
# bounded set of (batch, frame-bucket) shapes — variable arrival batch sizes otherwise trigger
# a fresh multi-second compile per novel size (measured: p95 10-16s without this)
BATCH_GRID = (1, 2, 4, 8, 16, 32, 48)


def _grid_size(k: int) -> int:
    for g in BATCH_GRID:
        if g >= k:
            return g
    return BATCH_GRID[-1]


class TurboServeEngine:
    def __init__(self, model_dir: str | None = None, adaptive: bool | None = None,
                 exec_batch: int | None = None, compile: bool | None = None,
                 compile_mode: str | None = None, device: str = "cuda"):
        from fast import FastGraniteASR
        model_dir = model_dir or os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(_ROOT), "ref"))
        self.adaptive = (os.environ.get("ADAPTIVE", "1") == "1") if adaptive is None else adaptive
        self.exec_batch = int(os.environ.get("EXEC_BATCH", "48")) if exec_batch is None else exec_batch
        compile = (os.environ.get("COMPILE", "1") == "1") if compile is None else compile
        compile_mode = os.environ.get("COMPILE_MODE") or compile_mode
        self.asr = FastGraniteASR(model_dir, device=device, compile=compile,
                                  compile_mode=compile_mode, adaptive=self.adaptive)
        self._fn = self.asr.transcribe_adaptive if self.adaptive else self.asr.transcribe
        # one engine = one CUDA context/stream: serialize GPU entry, overlap happens across
        # replicas (run 2 replicas per GPU to hide host prep/decode — measured 1.4-4.5% e2e)
        self._gpu_lock = threading.Lock()

    def warmup(self, seconds=(2, 5, 10, 20, 30), batch_sizes=None) -> None:
        """Populate compile caches for the shape buckets a serving workload actually hits."""
        if batch_sizes is None:
            batch_sizes = [g for g in BATCH_GRID if g <= self.exec_batch]
        for bs in batch_sizes:
            for s in seconds:
                wav = torch.zeros(int(s * SR), dtype=torch.float32)
                with self._gpu_lock:
                    self._fn([wav] * bs)

    def transcribe_batch(self, waveforms: list) -> list[str]:
        """Duration-sorted chunked execution; returns texts in the original order."""
        wavs = [torch.as_tensor(np.asarray(w, dtype=np.float32)) for w in waveforms]
        n = len(wavs)
        out: list[str | None] = [None] * n

        long_idx = [i for i, w in enumerate(wavs) if w.shape[-1] > MAX_CLIP_S * SR]
        short_idx = [i for i in range(n) if i not in set(long_idx)]

        order = sorted(short_idx, key=lambda i: wavs[i].shape[-1])
        for c0 in range(0, len(order), self.exec_batch):
            idx = order[c0:c0 + self.exec_batch]
            batch = [wavs[i] for i in idx]
            pad = _grid_size(len(batch)) - len(batch)
            if pad:
                batch = batch + [batch[-1]] * pad   # pad with copies; outputs discarded
            with self._gpu_lock:
                texts = self._fn(batch)
            for i, t in zip(idx, texts[:len(idx)]):
                out[i] = t

        for i in long_idx:
            with self._gpu_lock:
                out[i] = self.asr.transcribe_long([wavs[i]], max_s=MAX_CLIP_S)[0]
        return out  # type: ignore[return-value]


def decode_wav_bytes(data: bytes) -> torch.Tensor:
    """WAV/FLAC/OGG bytes -> mono float32 16 kHz tensor (soundfile + librosa resample)."""
    import soundfile as sf
    arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    arr = arr.mean(axis=1)
    if sr != SR:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
