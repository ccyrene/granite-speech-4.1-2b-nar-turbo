# `best/` — fastest measured bench config (Granite Speech 4.1 2B NAR, H100)

One-command reproduction of the session-C record run:

```bash
python script/best_run.py                    # subset-500, defaults = record run
python script/best_run.py --max-samples 64   # quick smoke test
python script/best_run.py --split test.other --max-samples 0   # full set
```

## The config

| what | value |
|---|---|
| levers | `compile-enc, compile-proj, compile-llm, texthead, adaptive, flexattn, encattn, encdense, freeze, convkernel` |
| env | `FRAME_GRID=128 EXEC_BATCH=48 ENC_COMPILE_MODE=max-autotune-no-cudagraphs` |
| batch | logical 128, executed in length-trimmed chunks of ≤48 |

`convkernel` = `granite::dwconv_silu` custom op (fused depthwise conv + bias + SiLU after
BN fold, `models/granite_speech_nar/conv_kernel.py`) — the only hand kernel that beats
torch/inductor on H100 for this model. Requires `cuda-python` + the nvidia CUDA wheels
(NVRTC); silently falls back to the torch path if unavailable (`convkernel(0)` in the log
means fallback — check, don't assume).

## Measured results (H100 SXM 80GB, torch 2.9.1+cu130, LibriSpeech)

| run | n | RTFx e2e | RTFx model | WER jiwer/kaldi |
|---|---:|---:|---:|---|
| subset test.clean (record) | 500 | **3934.7** | **4025.7** | 1.27/1.09 |
| full test.clean gate | 2620 | 3485.8 | 3496.0 | 1.42/1.19 |
| full test.other gate | 2939 | 3276.7 | 3287.0 | 2.81/2.52 |

Same-instance baselines without `convkernel`: b48 champion 3887.1, b128ex48 3878.5.
Session progression (subset): A 3247.6 → B 3874.9 → C 3934.7.

Accuracy class: WER-gated (encattn + BN fold are not bit-exact). Transcripts verified exact
vs reference-implementation goldens; full-set gates within the Δjiwer ≤ +0.05 budget.

## Notes

- `EXEC_BATCH≤48` is REQUIRED with max-autotune at b≥64 (inductor autotune hits
  cudaErrorIllegalAddress at the larger GEMM shapes on torch 2.9.1). 48 is the
  optimum — 40 measured −1.7%.
- b≤48 without EXEC_BATCH: use `--batch 48 --exec-batch 0`... keep EXEC_BATCH=48; a plain
  b48 run is the same execution shape minus the logical-batch prep, measured 3921.3.
- First run compiles ~5-10 min (FRAME_GRID buckets × autotune); TORCHINDUCTOR_CACHE_DIR
  persists it.

Measured records with this exact config: H200 3951.5 / H100 3934.7 / A100 1468.4 RTFx e2e
(see the top-level README for the full tables and WER verification).
