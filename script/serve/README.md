# serve/ — Ray Serve vs Triton, same engine

Both backends wrap the same `serve/engine.py` (`TurboServeEngine`): duration-sorted execution
chunks of ≤`EXEC_BATCH`, frame-grid compile bucketing, shape-grid warmup, and long-form (>30 s)
chunk+merge routing. The only thing that differs is the request scheduling layer — which is
exactly what you want to compare.

## Quickstart

### Ray Serve

```bash
pip install "ray[serve]" httpx
MODEL_DIR=ref PYTHONPATH=script serve run serve.ray_app:app                 # single replica, 1 GPU
# pack 2 replicas per GPU to hide host prep/decode:
NUM_REPLICAS=2 GPUS_PER_REPLICA=0.5 PYTHONPATH=script serve run serve.ray_app:app

curl -X POST --data-binary @sample.wav http://127.0.0.1:8000/transcribe
```

### Triton

```bash
docker build -t granite-asr-triton -f serve/triton/Dockerfile .
docker run --gpus 1 --rm -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf_snapshot:/models/granite -e MODEL_DIR=/models/granite \
  -v inductor-cache:/root/.inductor -e TORCHINDUCTOR_CACHE_DIR=/root/.inductor \
  granite-asr-triton
```

### Load test (same client for both — apples to apples)

```bash
python script/serve_client.py --backend ray    --synth -c 32 -n 512
python script/serve_client.py --backend triton --synth -c 32 -n 512   # pip install tritonclient[http]
# or point --wav-dir at real audio
```

Reports served-audio RTFx (audio seconds ÷ wall seconds), latency p50/p95/p99.

## How each backend batches

| | Ray Serve | Triton |
|---|---|---|
| Dynamic batching | `@serve.batch(max_batch_size=16, batch_wait_timeout_s=0.05)` — batches concurrent awaits | `dynamic_batching` scheduler (max_batch_size 16), 50 ms queue delay, ragged AUDIO input |
| Length handling | engine sorts by duration inside the batch | same engine, same sort |
| 2-per-GPU packing | `GPUS_PER_REPLICA=0.5` | `instance_group { count: 2 }` in config.pbtxt |
| Warmup | in `__init__` before replica is Ready | in `initialize()` before model is Ready |

## Measured (A100 PCIE 40GB, 2026-07-03)

Real LibriSpeech clips (~25 s each), 256 requests, both backends capped at batch 16 over the
same engine (`ADAPTIVE=1`, `EXEC_BATCH=16`, warm inductor cache), client on the same box:

| | concurrency | served-RTFx | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| **Triton** | 8 | **720** | **267 ms** | **320 ms** | 399 ms |
| **Triton** | 32 | **529** | 1008 ms | 3863 ms | 4836 ms |
| Ray Serve | 8 | 287 | 723 ms | 942 ms | 2317 ms |
| Ray Serve | 32 | 332 | 2385 ms | **2816 ms** | 2886 ms |

Reading: Triton's binary tensor transport + C++ scheduler wins throughput and low-load latency
decisively; Ray's tail is tighter under saturation and the stack is far easier to debug/extend.
Transport asymmetry to keep in mind: the Triton client ships decoded fp32 PCM (decode paid
client-side), while the Ray ingress accepts WAV bytes and pays soundfile decode server-side.

### Hard-won findings baked into this directory

1. **Batch-dim bucketing is mandatory for serving** (`engine.py` `BATCH_GRID`): arrival batches
   vary 1..N, and every novel (batch, frame-bucket) shape triggers a fresh multi-second
   torch.compile — measured p95 of 10-16 s before the fix, 0.3-2.8 s after. The bench harness
   never sees this because it always runs fixed-size batches.
2. **Match the batcher cap to `EXEC_BATCH`**: Triton at `max_batch_size: 48` over a 16-chunk
   engine executed 3 sequential chunks per group → 3× tail latency. Capped at 16 → p95 dropped
   from 8.2 s to 0.32-3.9 s.
3. `tritonclient.http` is gevent-based — never share/use it across OS threads (deadlock);
   use one client + `async_infer` sliding window (see `script/serve_client.py`).
4. With `max_batch_size > 0` the first dim of every input is the batch dim: send AUDIO as
   `[1, T]`, not `[T]` (the server rejects with "batch-size must be <= N").
5. `from __future__ import annotations` breaks FastAPI param binding inside `serve.ingress`
   classes (Request becomes a required query param → every call 422s). Keep it out of ray_app.
6. NGC Triton image quirks (repaired in the Dockerfile): DALI's bundled wheel dir shadows
   `attrs`/`packaging` (pip thinks they exist; non-login shells can't import them), and the
   image ships protobuf 7.x while ray-serve needs the pre-6 descriptor API → pin `~=5.29`.

## When to pick which

**Ray Serve** — you own the scheduling loop in Python: easiest place to extend the pipeline
(prep/decode overlap, priority lanes, session affinity, custom admission control), replicas
scale with a one-line config, and the deployment is plain Python — no container required.
Best default for this model.

**Triton** — you already run a Triton fleet and want its ops surface (Prometheus metrics,
model repository versioning, ensembles with other models, gRPC + shared-memory transport,
KServe protocol clients). The python backend means the model code is identical — you pay
container/config complexity for the ops features.

**Neither vLLM nor SGLang fits**: this model is non-autoregressive (single forward pass, no
KV cache, no decode loop) — continuous batching and paged attention have nothing to grab.

## Knobs that matter (both backends)

- `EXEC_BATCH` (default 48): keep ≤48 with max-autotune — b≥64 hits a known Inductor
  autotune failure class on torch 2.9.1.
- `ADAPTIVE=1`: CTC-first routing, ≤+0.08 WER points, large throughput win. `0` = strictly
  lossless path.
- `TORCHINDUCTOR_CACHE_DIR` + `GRANITE_PTX_CACHE` on a persistent volume: first boot pays
  5-10 min of autotune, every boot after that is warm.
- Batch window (50 ms default): raise it under bursty low-QPS traffic to form fuller
  batches; lower it when p95 latency matters more than throughput.
- Two engine instances per GPU (~7-13 GB VRAM each) overlap one instance's host work with
  the other's GPU work — this recovers the measured host-side e2e/model gap (1.4% on A100
  up to 4.5% on H200).
