# serve/ — Ray Serve vs Triton, same engine

Both backends wrap the same `serve/engine.py` (`TurboServeEngine`): duration-sorted execution
chunks of ≤`EXEC_BATCH`, frame-grid compile bucketing, shape-grid warmup, and long-form (>30 s)
chunk+merge routing. The only thing that differs is the request scheduling layer — which is
exactly what you want to compare.

## Quickstart

### Ray Serve

```bash
pip install "ray[serve]" httpx
MODEL_DIR=ref serve run serve.ray_app:app                 # single replica, 1 GPU
# pack 2 replicas per GPU to hide host prep/decode:
NUM_REPLICAS=2 GPUS_PER_REPLICA=0.5 serve run serve.ray_app:app

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
python scripts/serve_client.py --backend ray    --synth -c 32 -n 512
python scripts/serve_client.py --backend triton --synth -c 32 -n 512   # pip install tritonclient[http]
# or point --wav-dir at real audio
```

Reports served-audio RTFx (audio seconds ÷ wall seconds), latency p50/p95/p99.

## How each backend batches

| | Ray Serve | Triton |
|---|---|---|
| Dynamic batching | `@serve.batch(max_batch_size=48, batch_wait_timeout_s=0.05)` — batches concurrent awaits | `dynamic_batching` scheduler, 50 ms queue delay, ragged AUDIO input |
| Length handling | engine sorts by duration inside the batch | same engine, same sort |
| 2-per-GPU packing | `GPUS_PER_REPLICA=0.5` | `instance_group { count: 2 }` in config.pbtxt |
| Warmup | in `__init__` before replica is Ready | in `initialize()` before model is Ready |

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
