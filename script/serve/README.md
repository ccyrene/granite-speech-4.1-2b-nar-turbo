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

### Transport + scaling matrix (A100 SXM4, 2026-07-03) — gRPC / system-shm / instance_group=2

Follow-up run answering the two questions the table above left open: does gRPC / system
shared-memory beat HTTP, and does a second engine per GPU (`instance_group: 2`) recover the
host-side gap. Same engine + client, real LibriSpeech (64 clips, 256 requests, warm cache,
`ADAPTIVE=1`, `EXEC_BATCH=16`, client on-box). All rows `errors=0`.

| config | concurrency | served-RTFx | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| Triton x1 · http | 8 | 250 † | 266 ms | 5744 ms † | 6061 ms |
| Triton x1 · http | 32 | 116 † | 8478 ms | 14303 ms † | 14340 ms |
| Triton x1 · grpc | 8 | 654 | 243 ms | 591 ms | 637 ms |
| Triton x1 · grpc | 32 | 568 | 1320 ms | 2503 ms | 2510 ms |
| Triton x1 · grpc+shm | 8 | 653 | 246 ms | 665 ms | 782 ms |
| **Triton x1 · grpc+shm** | **32** | **739** | 1098 ms | **2035 ms** | 2047 ms |
| Triton x2 (instance_group=2) · grpc | 8 | 367 | 285 ms | 1690 ms | 1823 ms |
| Triton x2 (instance_group=2) · grpc | 32 | 581 | 1044 ms | 3346 ms | 3349 ms |
| Triton x2 (instance_group=2) · grpc | 64 | 627 | 2286 ms | 4703 ms | 4790 ms |
| Ray Serve (same host) · http | 8 | 314 | 650 ms | 879 ms | 2251 ms |
| Ray Serve (same host) · http | 32 | 329 | 2381 ms | 2844 ms | 2883 ms |

† **HTTP ran first** (right after warmup), so it absorbed the residual-compile tail from novel
batch/frame buckets — its p95 (5.7–14.3 s) and throughput are a warmup-order artifact, not a
transport verdict. Its p50 @ c=8 (266 ms) matches the isolated warm HTTP run above (267 ms), so
per-request cost is transport-agnostic; treat the http rows as cold and grpc/shm/x2/ray as warm.

Reading:
- **Best serving config: one Triton instance, gRPC + system shared-memory → 739 RTFx @ c=32**
  (p50 1.1 s / p95 2.0 s), ~653 @ c=8.
- **Shared memory earns its keep at high concurrency**: at c=32 grpc+shm (739) beats plain grpc
  (568) by +30 %; at c=8 they tie (~653). The win scales with aggregate payload size — zero-copy
  vs. a socket copy of ~1.6 MB fp32 PCM per request.
- **`instance_group: 2` does NOT help this model — it hurts.** Two engines share one A100's SMs
  and VRAM: at c=8 x2 (367) is 44 % *below* x1 grpc (654), and even at c=64 x2 (627) never
  reaches x1's warm best (739). This single-pass NAR model is compute-bound and one engine
  already saturates the GPU; a second one only adds contention. Refutes the earlier
  "2-per-GPU recovers the host-side gap" hypothesis on A100.

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
- Two engine instances per GPU (`instance_group: 2` / `GPUS_PER_REPLICA=0.5`) was meant to
  overlap one instance's host work with the other's GPU work, but **measured on A100 it lost to
  a single engine at every concurrency** (see the transport + scaling matrix) — this
  compute-bound single-pass model already saturates one GPU, so a second engine only adds
  SM/VRAM contention. Close the host-side e2e/model gap with prep/decode overlap *inside* one
  engine instead.
