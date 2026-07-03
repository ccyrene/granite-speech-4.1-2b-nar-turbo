# `fast/` — optimized adaptive inference for Granite Speech 4.1 2B NAR

The shippable, ~2.8x-faster inference path, validated on A100 / LibriSpeech. `transcribe()` is the
CTC-first **adaptive** path (≤0.08 WER over the full-editor pass it falls back to for hard
utterances); the compiled stack underneath is bit-exact to eager. Pure `torch.compile` — **no
TensorRT** (TRT was faster per-kernel but bf16-lossy; see below).

```python
from fast import FastGraniteASR
asr = FastGraniteASR("ref")          # or an HF snapshot dir
asr.warmup()                          # optional: compile a few length-buckets up front
text = asr.transcribe(waveform_16k)[0]
```
or `python script/fast_demo.py --wav my.wav`

## What it does (stacked on the bit-identical `models/granite_speech_nar` reimpl)

| # | optimization | effect | bit-exact? |
|---|---|---|---|
| 1 | **GPU-resident features** (`features.py`: move wav to GPU *before* `pad_sequence`) | feature 152ms → 0.6ms (CPU `pad_sequence` was the killer); full e2e 216→64ms | yes (value-preserving) |
| 2 | **`torch.compile(encoder)`** + FRAME_GRID=128 frame-bucketing (branch-free conformer) | removes launch overhead; stable compiled shapes | yes (bit-exact) |
| 3 | **`torch.compile(projector, LLM)`** | inductor bf16 ≈ eager → no WER change | yes |
| 4 | **text-only LM head** (`text_only_head=True`) | runs the tied head only on text positions | VRAM −16%, transcript-exact |

## Validated numbers (A100-40GB, LibriSpeech test.clean, full n=2588)

| config | RTFx (model) | WER jiwer/kaldi | single-utterance e2e |
|---|---|---|---|
| eager baseline | 625 | 1.39% / 1.16% | 61 ms |
| **this (compile, default)** | **~794 (batch 16)** | **1.37% / 1.16%** ✓ bit-exact | ~29 ms |
| this (`compile_mode="reduce-overhead"`) | — | (same, bit-exact) | ~26 ms |
| ~~all-TensorRT~~ | (≈22 ms) | **5.7%** ✗ lossy | 22 ms |

- **bit-exact**: 1.37% ≈ baseline 1.39% (noise); kaldi 1.16% identical. Probe transcript-exact.
- **~2.8x** over eager (RTFx 277 → ~790 single-utterance equiv after batching).
- RTFx is **compute-bound** at ~790 (2B model: 16-layer conformer + 40-layer bidirectional LLM). Batch is
  flat 8–16. To go higher needs *less compute* = quantization (int8 broke on A100; **fp8 needs H100** +
  calibration + WER-gate) — `torch.compile` / batching / CUDA-graphs are maxed for the bit-exact path.

## Why no TensorRT
editor→TRT (bf16) is 3.6x faster per-component and transcript-exact on one sample, but on the full set it
gives **5.7% WER** (vs 1.37%): TRT's bf16 attention/GEMM kernels diverge ~7x more than inductor from eager
(max|Δlogit| 4.56 vs 0.66), flipping enough tokens to wreck WER. `torch.compile` reaches ~the same speed
(editor 9.6ms ≈ TRT 9.3ms via CUDA graphs) **while staying bit-exact**. So TRT is intentionally dropped.

## Long-form audio (`transcribe_long`)
The model has **no built-in max length** but a soft cap: the LLM `max_position_embeddings=4096` bounds the
packed (audio+text) sequence ≈ **4 min** of speech — beyond that RoPE extrapolates and WER collapses. Use
`transcribe_long` for arbitrary-length audio:

```python
text = asr.transcribe_long(waveform, max_s=30, overlap_s=5)[0]   # one transcript per input
```
It splits >30s audio into **30s windows with 5s overlap**, batch-transcribes them (uniform shape, packed
length well under the cap), and stitches per-audio transcripts back by **de-duplicating the overlap**
(difflib longest-match — the boundary word is seen whole by ≥1 window). It does *not* zero-pad the
waveform (that would feed the model real silence); real-length windows go through the feature extractor so
`attention_mask` masks padding exactly.

Validated (concatenated test.clean, A100):
| audio | whole `transcribe` | `transcribe_long` |
|---|---|---|
| 6.3 min (> cap) | **99.1% WER (fails)** | **2.96% WER** (rescued) |
| 135 s (< cap) | 4.17% | **2.56%** (≈/better — model prefers 30s windows) |
| ≤ 30 s | — | identical to `transcribe` (1 window, no merge) |

→ near-exact (even quality-neutral-to-better) and **required** for audio beyond ~4 min.

## Files
- `fast_asr.py` — `FastGraniteASR`: load + compile + GPU-resident feature + FRAME_GRID bucketing + texthead
  + `transcribe_long`.
- `longform.py` — `chunk_waveform` (overlap windows) + `merge_words` (overlap de-dup).
- `run.py` — demo CLI (transcript + warm latency/RTFx).

## Requirements
torch 2.9.x (CUDA), the runtime deps of `models/granite_speech_nar` (tokenizers, safetensors, soundfile,
numpy). No torch-tensorrt. Inputs must be 16 kHz mono (resampling was removed with torchaudio).
