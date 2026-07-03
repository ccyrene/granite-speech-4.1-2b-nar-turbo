# granite-speech-turbo

Hand-tuned pure-PyTorch inference stack for
[`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
(non-autoregressive CTC-editing ASR, [NLE architecture](https://arxiv.org/abs/2603.08397)) —
**~3× the official HuggingFace implementation head-to-head, ~2.2× the model card's published
RTFx, with WER verified identical to the reference.**

| GPU | RTFx (e2e) | RTFx (model-only) | WER j/k | VRAM |
|---|---:|---:|---|---:|
| H200 SXM | **3951.5** | 4135.7 | 1.29 / 1.10 | 13.1 GB |
| H100 SXM | **3934.7** | 4025.7 | 1.27 / 1.09 | 13.0 GB |
| A100 PCIE 40GB | **1468.4** | 1489.1 | 1.27 / 1.10 | 13.1 GB |
| *(reference impl, A100, documented path)* | *487.0* | — | *1.38 / 1.15* | *6.0 GB* |

RTFx = total audio seconds ÷ wall-clock inference seconds (mel + model + decode), LibriSpeech
test.clean, 500-clip subset, bf16, batch 128 (executed in chunks of 48). IBM's model card reports
~1,820 RTFx on H100 at batch 128.

## WER — verified, not assumed

**Head-to-head vs the official implementation** (same audio, same scorer, same GPU):

| dataset (full test set) | this repo | official impl | 
|---|---:|---:|
| LibriSpeech clean | 1.38 | 1.38 |
| LibriSpeech other | 2.76 | 2.79 |
| AMI | 7.98 | 8.09 |
| SPGISpeech (39,341 utts) | 3.49 | 3.48 |

**Vs the model card** (same stated methodology — greedy, bf16, jiwer + Whisper EnglishTextNormalizer):

| dataset | model card | this repo (lossless) | this repo (adaptive) |
|---|---:|---:|---:|
| LibriSpeech clean | 1.29 | 1.38 | 1.40 |
| LibriSpeech other | 2.75 | 2.76 | 2.81 |
| AMI | 7.91 | 7.98 | 8.03 |
| Earnings-22 | 8.48 | **8.48** | **8.48** |
| GigaSpeech | 10.12 | 10.22 | 10.21 |
| SPGISpeech | 3.04 | 3.49* | 3.49* |
| VoxPopuli | 5.83 | 5.90 | 5.97 |

*The SPGISpeech gap lives in the scoring pipeline, not the model: the official implementation
scores 3.48 through this repo's scorer (see head-to-head above). WER absolute numbers are only
comparable within one scoring pipeline — even IBM's own two published numbers disagree per set
(e.g. GigaSpeech 10.12 model card vs 8.67 leaderboard).

## What makes it fast

- **`torch.compile` max-autotune with frame-grid bucketing** — audio lengths are bucketed to a
  fixed grid (`FRAME_GRID=128`) so Inductor compiles a bounded shape set instead of recompiling
  per length.
- **Chunked logical batching (`EXEC_BATCH`)** — a logical batch of 128 is executed as
  length-trimmed chunks of ≤48: kills padding waste and avoids an Inductor autotune failure class
  at b≥64. This is what makes large-batch beat small-batch.
- **One hand-written CUDA kernel that beats the compiler** — fused depthwise-conv(k15)+bias+SiLU
  after BatchNorm folding (`models/granite_speech_nar/conv_kernel.py` + `cuda_kernels/conv1d.py`),
  registered as a `torch.library` custom op: one opaque node, fullgraph-safe, zero graph breaks.
  Microbench +19–32% across A100/H100/H200; end-to-end +0.6–1.5%. Falls back to the eager path
  automatically if NVRTC compilation or the numerical test-fire fails (look for `convkernel(16)`
  vs `convkernel(0)` in logs).
- **Confidence-routed adaptive inference** (`adaptive` lever, `configs/routing.yaml`) — easy
  utterances exit via the CTC hypothesis; only hard ones pay for the full LLM-editor pass.
  ≤ +0.08 WER points on every set measured (free on Earnings-22 and GigaSpeech).
- **FlexAttention + restructured encoder attention** (`flexattn`, `encattn`, `encdense` levers)
  for the Conformer block-attention and relative-position path.

### What did **not** work (measured, so you don't have to)

- Hand-written GEMM / attention kernels: lose **10–70×** to cuBLAS / SDPA. The losing kernels are
  kept in `cuda_kernels/` for reference.
- GLU-fused depthwise conv: **+17% on RTX 3060, −45% on H100** — memory-bound wins flip to
  ALU-bound losses on high-bandwidth parts. Always re-gate microbenches on the target GPU
  (`python -m cuda_kernels.conv1d`).
- Chunked 100k-vocab text-head argmax: same HBM bytes, more launches (a real fix needs a fused
  GEMM epilogue).

## Install

```bash
pip install -r requirements.txt          # torch 2.9.1 + deps
# NOTE (2026-07): the cu130 pip index no longer lists torch 2.9.x — install from the cu128 index
# or pin the direct wheel URL; a torch other than 2.9.x is an unproven stack for these numbers.
```

## Quickstart

```bash
# 1) fetch the model (≈4.3 GB) and symlink it to ./ref
python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("ibm-granite/granite-speech-4.1-2b-nar",
      allow_patterns=["config.json","model.safetensors","tokenizer.json","*.wav"]))
EOF
ln -s <printed_path> ref

# 2) reproduce the record run (subset-500, all levers, b128/exec48)
python scripts/best_run.py

# 3) full-set WER gates
python scripts/best_run.py --variant gate_clean --split test.clean --max-samples 0
python scripts/best_run.py --variant gate_other --split test.other --max-samples 0
```

Direct harness invocation (all levers explicit):

```bash
FRAME_GRID=128 EXEC_BATCH=48 ENC_COMPILE_MODE=max-autotune-no-cudagraphs \
python scripts/bench_asr.py \
  --levers compile-enc,compile-proj,compile-llm,texthead,adaptive,flexattn,encattn,encdense,freeze,convkernel \
  --model-dir "$(readlink -f ref)" --config librispeech --split test.clean \
  --batch 128 --max-samples 500 --no-probe
```

## Repo layout

```
models/granite_speech_nar/   pure-PyTorch reimplementation (encoder/projector/LLM editor,
                             adaptive routing, BN folding, custom-op integration)
cuda_kernels/                the winning fused dwconv+bias+SiLU NVRTC kernel
                             (self-benchmark: python -m cuda_kernels.conv1d)
fast/                        compile-friendly serving wrapper (FastGraniteASR, 30s chunking)
best/                        one-command record-run configuration
serve/                       production serving: Ray Serve + Triton backends over one shared
                             engine, with a unified load-test client (see serve/README.md)
scripts/bench_asr.py         the benchmark harness behind every number in this README
configs/routing.yaml         adaptive-routing thresholds
```

## Methodology notes

- WER: jiwer (corpus-level) on Whisper-`EnglishTextNormalizer`-normalized text, kaldialign as a
  second opinion; empty normalized references dropped. Identical scorer applied to both this repo
  and the reference implementation for every claim above.
- Run-to-run noise on one instance is ±0.6% RTFx; sub-1% deltas need repeats.
- Numbers were measured on Vast.ai instances (driver CUDA 13.0, torch 2.9.1+cu130).

## License & acknowledgments

Apache-2.0 (matching the upstream model license). Model weights & architecture:
[IBM Granite](https://huggingface.co/ibm-granite) — this repo contains no weights.
Evaluation data: [ESB / Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
