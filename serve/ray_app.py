"""Ray Serve backend.

    pip install "ray[serve]"
    serve run serve.ray_app:app          # from the repo root; MODEL_DIR=ref by default

Env knobs:
    MODEL_DIR        path to the HF snapshot (default: ./ref)
    ADAPTIVE=1       CTC-first adaptive routing (0 = strictly lossless path)
    NUM_REPLICAS=1   replicas total
    GPUS_PER_REPLICA=1.0   set 0.5 to pack 2 replicas per GPU — hides host prep/decode
                           (the measured e2e/model gap: 1.4% A100 -> 4.5% H200)
    MAX_BATCH=48 BATCH_WAIT_S=0.05   dynamic batching window
    WARMUP=1         run the shape-grid warmup before serving traffic

Request:  POST /transcribe  with audio bytes (wav/flac/ogg) as the body
Response: {"text": ..., "audio_s": ..., "latency_ms": ...}
"""
from __future__ import annotations

import asyncio
import os
import time

from fastapi import FastAPI, Request
from ray import serve

from serve.engine import SR, TurboServeEngine, decode_wav_bytes

api = FastAPI()


@serve.deployment(
    num_replicas=int(os.environ.get("NUM_REPLICAS", "1")),
    ray_actor_options={"num_gpus": float(os.environ.get("GPUS_PER_REPLICA", "1.0"))},
)
@serve.ingress(api)
class GraniteASR:
    def __init__(self):
        self.engine = TurboServeEngine()
        if os.environ.get("WARMUP", "1") == "1":
            self.engine.warmup()

    @serve.batch(max_batch_size=int(os.environ.get("MAX_BATCH", "48")),
                 batch_wait_timeout_s=float(os.environ.get("BATCH_WAIT_S", "0.05")))
    async def _batched(self, wavs: list) -> list[str]:
        # the whole point: requests that arrive within the window are executed as ONE
        # duration-sorted chunked batch instead of many padded singles
        return await asyncio.to_thread(self.engine.transcribe_batch, wavs)

    @api.post("/transcribe")
    async def transcribe(self, request: Request):
        t0 = time.perf_counter()
        wav = decode_wav_bytes(await request.body())
        text = await self._batched(wav)
        return {"text": text, "audio_s": round(wav.shape[-1] / SR, 3),
                "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)}

    @api.get("/health")
    async def health(self):
        return {"ok": True}


app = GraniteASR.bind()
