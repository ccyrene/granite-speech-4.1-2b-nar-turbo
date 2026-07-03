"""Open ASR Leaderboard-style IN-PROCESS benchmark for the PURE-TORCH granite-opt reimpl.

Measures RTFx (= total_audio_sec / total_inference_sec) AND WER (Whisper EnglishTextNormalizer, the
leaderboard's normalizer) on standard test sets, batched, in-process (NO serving layer). Run baseline
then add ONE lever at a time to see how much each step buys in RTFx while staying WER-lossless and
(for the exact levers) bit-identical to the eager baseline.

This is the granite-opt counterpart of granite-nar-handcraft/bench_asr.py, but it drives the pure-torch
`models.granite_speech_nar` pipeline (no transformers in the model path; transformers is used ONLY for
the WER normalizer).

Levers (comma list via --levers; cumulative ablation = add one each run):
  compile-enc   torch.compile(model.encoder,        dynamic=True)
  compile-proj  torch.compile(model.projector,      dynamic=True)
  compile-llm   torch.compile(editor, dynamic=True) -> the 40-layer LLM "editor" (~49% of compute;
                this is the component whose local TRT/compile froze the 6 GB box)
  texthead      run the tied LM head ONLY on the text segments (skip the audio-row full-vocab GEMM);
                argmax/transcript-exact vs baseline on the single-sample path
  int8          torchao dynamic W8A8 on encoder+editor block Linears (A100 sm_80 -> INT8, no FP8).
                Needs compile to pay off -> pair with compile-* . WER-gated (not bit-identical).

Usage:
  PY=.venv/bin/python
  $PY scripts/bench_asr.py --variant baseline --model-dir <hf_snapshot> --tokenizer <hf_snapshot>/tokenizer.json \
      --config librispeech --split test.clean --batch 16 --max-samples 0
Results append to results/bench_asr.json with a printed table.
"""
from __future__ import annotations

import argparse, json, math, os, time
from collections import Counter
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

# vast.ai containers expose ALL host cores to torch while the cgroup grants few; the small CPU ops on
# the critical path (pad_sequence, mask build, decode) get catastrophic oversubscription. Cap threads.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.granite_speech_nar import load_model, MelFeatureExtractor, SpeechTokenizer, ASROutput  # noqa: E402

DEV = "cuda"
FRAME_GRID = int(os.environ.get("FRAME_GRID", "0"))   # 0 = off; >0 buckets encoder frames for compiled enc
SEQ_GRID = int(os.environ.get("SEQ_GRID", "0"))        # 0 = off; >0 buckets the adaptive editor packed length (Phase 7)
EXEC_BATCH = int(os.environ.get("EXEC_BATCH", "0"))    # 0 = off; >0 splits a logical batch before inference
COMPILE_MODE = os.environ.get("COMPILE_MODE") or None  # None=default; "max-autotune" / "reduce-overhead" (CUDA graphs)

# CUTLASS (or other) inductor GEMM backend. Set GEMM_BACKENDS="ATEN,TRITON,CUTLASS" + COMPILE_MODE=max-autotune
# to let inductor autotune CUTLASS tensor-core GEMMs for the model's Linear shapes (A100 sm_80: bf16/tf32/int8).
# Needs the `nvidia-cutlass` package; CUTLASS_DIR overrides where inductor finds the CUTLASS source.
GEMM_BACKENDS = os.environ.get("GEMM_BACKENDS", "").strip()
if GEMM_BACKENDS:
    import torch._inductor.config as _indcfg
    _indcfg.max_autotune_gemm_backends = GEMM_BACKENDS
    _cutlass_dir = os.environ.get("CUTLASS_DIR", "").strip()
    if _cutlass_dir and hasattr(_indcfg, "cuda"):
        _indcfg.cuda.cutlass_dir = _cutlass_dir


def configure_dynamo_limits(limit: int = 64):
    """Raise graph-cache/recompile limits before any torch.compile call."""
    import torch._dynamo as _dynamo
    cfg = _dynamo.config
    if hasattr(cfg, "cache_size_limit"):
        cfg.cache_size_limit = max(cfg.cache_size_limit, 128)
    if hasattr(cfg, "recompile_limit"):
        cfg.recompile_limit = max(cfg.recompile_limit, limit)
    if hasattr(cfg, "accumulated_recompile_limit"):
        cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, limit)


def _jsonable(v):
    """Return JSON-strict values; json.dump otherwise writes bare NaN tokens."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# --------------------------------------------------------------------------------------------------- #
# Levers
# --------------------------------------------------------------------------------------------------- #
def apply_int8(model):
    """torchao dynamic W8A8 on the repeated block Linears of encoder + editor (the GEMM bulk). A100
    (sm_80) has INT8 tensor cores, no FP8. Skips odd/argmax-sensitive heads. Needs compile to fuse."""
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    cfg = Int8DynamicActivationInt8WeightConfig()

    def filt(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and "layers." in fqn and min(mod.weight.shape) >= 64

    n = (sum(1 for fqn, m in model.encoder.named_modules() if filt(m, fqn))
         + sum(1 for fqn, m in model.language_model.named_modules() if filt(m, fqn)))
    quantize_(model.encoder, cfg, filter_fn=filt)
    quantize_(model.language_model, cfg, filter_fn=filt)
    return n


def apply_fp8(model):
    """torchao dynamic fp8 (e4m3) W8A8 on the repeated block Linears of encoder + editor. H100 (sm_90)
    has fp8 tensor cores (~2x bf16 GEMM); far more accurate than int8 (per-row dynamic scaling). Skips
    odd/argmax-sensitive heads (same filter as int8). Needs compile to fuse the fp8 GEMMs."""
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    try:
        from torchao.quantization import PerRow
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    except Exception:
        cfg = Float8DynamicActivationFloat8WeightConfig()   # default granularity

    def filt(mod, fqn):
        return isinstance(mod, torch.nn.Linear) and "layers." in fqn and min(mod.weight.shape) >= 64

    n = (sum(1 for fqn, m in model.encoder.named_modules() if filt(m, fqn))
         + sum(1 for fqn, m in model.language_model.named_modules() if filt(m, fqn)))
    quantize_(model.encoder, cfg, filter_fn=filt)
    quantize_(model.language_model, cfg, filter_fn=filt)
    return n


def transcribe_text_head(model, input_features, attention_mask=None, encoder_lengths=None):
    """Lever `texthead`: identical to model.transcribe() EXCEPT the tied LM head runs only on the text
    segments (the audio-row full-vocab GEMM is skipped). argmax/transcript-exact vs baseline.
    Uses the P2 batched machinery (host lengths, batched packing, vectorized collapse); keeps the
    /logits_scaling division because this path RETURNS logits (the identity probe compares them)."""
    from models.granite_speech_nar import attn_backends
    from models.granite_speech_nar.asr import _collapse_ids_flat, _preds_to_host
    cfg = model.config
    enc_out = model.encoder(input_features, attention_mask=attention_mask, output_hidden_states=True)
    enc_lens = model._host_lengths(input_features, attention_mask, encoder_lengths)
    pool_window = model.encoder.config.bpe_pooling_window
    bpe_lengths = [-(-l // pool_window) for l in enc_lens]
    ctc_token_ids = model._ctc_collapse_decode(enc_out.logits, bpe_lengths)
    audio_embeds = model._project_audio(
        [enc_out.all_hidden_states[idx] for idx in cfg.encoder_layer_indices]
    )
    downsample = model.projector.config.downsample_rate
    audio_lengths = [l // downsample for l in enc_lens]
    seg_audio = [(i, 0, audio_lengths[i]) for i in range(len(audio_lengths))]
    flat_embeds, flat_pos, layout = model._pack_editor_segments(audio_embeds, seg_audio, ctc_token_ids)
    text_lengths = [t for (_a, t) in layout]
    seg = [l for at in layout for l in at]
    editor_mask = attn_backends.build_editor_block_mask_from_lengths(seg, flat_embeds.device, flat_embeds.shape[1])
    hidden = model.language_model.model(
        inputs_embeds=flat_embeds, position_ids=flat_pos, attention_mask=editor_mask
    ).squeeze(0)  # base, no head
    text_hidden = torch.cat(list(hidden.split(seg)[1::2]), dim=0)
    text_logits = F.linear(text_hidden, model.language_model.model.embed_tokens.weight) / model.language_model.logits_scaling
    logits_per_sample = list(text_logits.split(text_lengths))
    preds, _counts, _kept = _collapse_ids_flat(text_logits.argmax(-1), text_lengths, cfg.blank_token_id)
    return ASROutput(preds=preds, preds_host=_preds_to_host(preds), logits=logits_per_sample)


def build(model, levers):
    """Apply levers in the correct order: int8 (quantize) -> compile (lowers int8) ; texthead / adaptive
    pick the transcribe fn. 'adaptive' = CTC-first routing (transcribe_adaptive, routing enabled), which
    internally honours texthead + SEQ_GRID; on the compiled path it reuses the same compiled enc/proj/llm.
    Kernels are toggled out-of-band via GRANITE_KERNEL_BACKEND (read at import by kernels.py)."""
    lv = {x.strip() for x in levers.split(",") if x.strip()}
    applied = []
    texthead = "texthead" in lv
    adaptive = "adaptive" in lv
    if "flexattn" in lv:      # editor FlexAttention block-diagonal (see attn_backends.py)
        os.environ["GRANITE_EDITOR_ATTN"] = "flex"
        applied.append("flexattn")
    if "encattn" in lv:       # conformer fused-SDPA attention (un-force MATH)
        os.environ["GRANITE_ENC_ATTN"] = "fused"
        applied.append("encattn")
    if "encdense" in lv:      # P3.1: dense BPE head -> single-graph encoder (transcript-safe)
        os.environ["GRANITE_ENC_DENSE_BPE"] = "1"
        applied.append("encdense")
    if "projslice" in lv:     # P2.3 ulp sub-lever: projector on need_editor rows only
        os.environ["GRANITE_PROJ_SLICE"] = "1"
        applied.append("projslice")
    if "freeze" in lv or os.environ.get("INDUCTOR_FREEZING", "").strip() == "1":  # P4.3
        import torch._inductor.config as _ind
        _ind.freezing = True
        applied.append("freeze")
    if "int8" in lv:
        applied.append(f"int8-w8a8({apply_int8(model)})")
    if "fp8" in lv:
        applied.append(f"fp8-w8a8({apply_fp8(model)})")
    if "convglu" in lv:       # Exp1b: GLU folded into the fused dwconv op (WER-gated)
        os.environ["GRANITE_CONV_KERNEL"] = "dwconv_silu_glu"
        from models.granite_speech_nar.conv_kernel import enable_dwconv_silu
        applied.append(f"convglu({enable_dwconv_silu(model, mode='dwconv_silu_glu')})")
    elif "convkernel" in lv:  # Exp1: fused dwconv+bias+SiLU custom op after BN-fold (WER-gated)
        os.environ["GRANITE_CONV_KERNEL"] = "dwconv_silu"
        from models.granite_speech_nar.conv_kernel import enable_dwconv_silu
        applied.append(f"convkernel({enable_dwconv_silu(model)})")
    if "textargmax" in lv:    # Exp2: chunked GEMM + running argmax text head (ulp-class)
        os.environ["GRANITE_TEXTHEAD_ARGMAX"] = "1"
        applied.append("textargmax")
    if "compile-enc" in lv:
        configure_dynamo_limits()
        enc_mode = os.environ.get("ENC_COMPILE_MODE", "").strip() or COMPILE_MODE  # P3.2/P3.3 rungs
        enc_fullgraph = os.environ.get("ENC_FULLGRAPH", "").strip() == "1"          # P3.1 check
        model.encoder = torch.compile(model.encoder, dynamic=(FRAME_GRID == 0), mode=enc_mode,
                                      fullgraph=enc_fullgraph)
        applied.append(f"compile-enc(dyn={FRAME_GRID==0}"
                       + (f",mode={enc_mode}" if enc_mode else "")
                       + (",fullgraph" if enc_fullgraph else "") + ")")
    if "compile-proj" in lv:
        configure_dynamo_limits()
        model.projector = torch.compile(model.projector, dynamic=True, mode=COMPILE_MODE)
        applied.append("compile-proj")
    if "compile-llm" in lv:
        configure_dynamo_limits()
        if texthead:
            model.language_model.model = torch.compile(model.language_model.model, dynamic=True, mode=COMPILE_MODE)
        else:
            model.language_model = torch.compile(model.language_model, dynamic=True, mode=COMPILE_MODE)
        applied.append("compile-llm")
    if texthead:
        applied.append("texthead")
    import os as _os
    if kernels_backend := _os.environ.get("GRANITE_KERNEL_BACKEND", "").strip().lower():
        applied.append(f"kernels={kernels_backend}")

    if adaptive:
        from models.granite_speech_nar.adaptive import load_adaptive_config, RoutingConfig
        try:
            routing, _v, _e = load_adaptive_config(os.path.join(ROOT, "configs", "routing.yaml"))
        except Exception:
            routing = RoutingConfig()
        routing.enabled = True
        applied.append(f"adaptive(seq_grid={SEQ_GRID})")
        transcribe = lambda inp: model.transcribe_adaptive(
            inp["input_features"], inp.get("attention_mask"),
            routing=routing, text_only_head=texthead, seq_grid=SEQ_GRID,
            encoder_lengths=inp.get("encoder_lengths"))
    elif texthead:
        transcribe = lambda inp: transcribe_text_head(model, inp["input_features"], inp.get("attention_mask"),
                                                      encoder_lengths=inp.get("encoder_lengths"))
    else:
        transcribe = lambda inp: model.transcribe(**inp)
    return transcribe, (applied or ["baseline"])


# --------------------------------------------------------------------------------------------------- #
# Data (esb-datasets-test-only-sorted, soundfile decode, sort by duration)
# --------------------------------------------------------------------------------------------------- #
def _detect_text_key(cols, text_key):
    if text_key:
        return text_key
    for k in ("text", "norm_text", "sentence", "transcription", "normalized_text"):
        if k in cols:
            return k
    raise ValueError(f"no text column in {cols}")


def _decode_audio(a):
    import io, soundfile as sf
    if isinstance(a, dict) and a.get("array") is not None:
        return np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
    src = io.BytesIO(a["bytes"]) if a.get("bytes") is not None else a["path"]
    arr, sr = sf.read(src, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def _row(r, tkey):
    arr, sr = _decode_audio(r["audio"])
    if sr != 16000:
        raise ValueError(f"expected 16kHz, got {sr}")
    return arr, len(arr) / 16000.0, r[tkey]


def load_data(path, config, split, text_key, max_samples):
    from datasets import load_dataset, Audio
    if max_samples and max_samples > 0:
        ds = load_dataset(path, config, split=split, streaming=True).cast_column("audio", Audio(decode=False))
        tkey = _detect_text_key(list(ds.features.keys()), text_key)
        rows = [_row(r, tkey) for r in ds.take(max_samples)]
    else:
        ds = load_dataset(path, config, split=split, verification_mode="no_checks").cast_column("audio", Audio(decode=False))
        tkey = _detect_text_key(ds.column_names, text_key)
        rows = [_row(ds[i], tkey) for i in range(len(ds))]
    rows.sort(key=lambda x: x[1])
    return rows, tkey


# --------------------------------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------------------------------- #
def wer_pct(refs, hyps):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    import jiwer
    NORM = EnglishTextNormalizer({})
    rn = [NORM(r) for r in refs]; hn = [NORM(h) for h in hyps]
    keep = [(r, h) for r, h in zip(rn, hn) if r.strip()]
    rn, hn = [r for r, _ in keep], [h for _, h in keep]
    wer_j = round(jiwer.wer(rn, hn) * 100, 2)
    wer_k = None
    try:
        from kaldialign import batch_error_rate
        wer_k = round(100 * batch_error_rate([tuple(r.split()) for r in rn],
                                             [tuple(h.split()) for h in hn], merge_compounds=True)["err_rate"], 2)
    except Exception as e:
        print(f"[bench] kaldialign WER unavailable ({e})")
    return wer_j, wer_k, len(keep)


# --------------------------------------------------------------------------------------------------- #
# Identity probe: compare lever transcribe vs eager-baseline on a few clips (BEFORE applying levers we
# capture the eager-baseline preds+logits on the probe batch; AFTER, we compare).
# --------------------------------------------------------------------------------------------------- #
@torch.inference_mode()
def probe_capture(model, fe, batch):
    wavs = [torch.from_numpy(a) for a, _, _ in batch]
    inp = fe(wavs, device=DEV)
    out = model.transcribe(**inp)
    return inp, [p.detach().clone() for p in out.preds], [l.detach().float().clone() for l in out.logits]


@torch.inference_mode()
def probe_compare(transcribe, inp, ref_preds, ref_logits):
    out = transcribe(inp)
    tok_match = sum(int(a.numel() == b.numel() and bool((a == b).all())) for a, b in zip(out.preds, ref_preds))
    maxd = 0.0
    if out.logits is None:                 # adaptive path returns preds only (fast-path has no logits)
        maxd = float("nan")
    else:
        for a, b in zip(out.logits, ref_logits):
            if a.shape == b.shape:
                maxd = max(maxd, float((a.float() - b).abs().max()))
            else:
                maxd = float("nan")
    return tok_match, len(ref_preds), maxd


# --------------------------------------------------------------------------------------------------- #
@torch.inference_mode()
def run(transcribe, fe, tok, rows, batch, warmup_batches=2, frame_grid=0):
    batches = [rows[i:i + batch] for i in range(0, len(rows), batch)]
    overlap = os.environ.get("GRANITE_NO_OVERLAP", "").strip() != "1"   # P2.5 kill-switch for A/B
    exec_batch = EXEC_BATCH if EXEC_BATCH > 0 else 0

    def prep(b):
        wavs = [torch.from_numpy(a) for a, _, _ in b]
        inp = fe(wavs, device=DEV)
        if frame_grid > 0:
            T = inp["input_features"].shape[1]
            fb = ((T + frame_grid - 1) // frame_grid) * frame_grid
            if fb > T:
                inp["input_features"] = F.pad(inp["input_features"], (0, 0, 0, fb - T))
                inp["attention_mask"] = F.pad(inp["attention_mask"], (0, fb - T))
        return inp

    def _batch_size(inp):
        for v in inp.values():
            if torch.is_tensor(v):
                return int(v.shape[0])
            if isinstance(v, (list, tuple)):
                return len(v)
        return 0

    def _slice_input(inp, start, end):
        B = _batch_size(inp)
        out = {}
        for k, v in inp.items():
            if torch.is_tensor(v) and v.shape[0] == B:
                out[k] = v[start:end]
            elif isinstance(v, list) and len(v) == B:
                out[k] = v[start:end]
            elif isinstance(v, tuple) and len(v) == B:
                out[k] = v[start:end]
            else:
                out[k] = v

        lens = out.get("encoder_lengths")
        if lens:
            max_len = max(int(x) for x in lens)
        elif out.get("attention_mask") is not None:
            max_len = int(out["attention_mask"].sum(dim=1).max().item())
        else:
            max_len = int(out["input_features"].shape[1])
        if frame_grid > 0:
            max_len = ((max_len + frame_grid - 1) // frame_grid) * frame_grid
        max_len = min(max_len, int(out["input_features"].shape[1]))
        out["input_features"] = out["input_features"][:, :max_len]
        if out.get("attention_mask") is not None:
            out["attention_mask"] = out["attention_mask"][:, :max_len]
        return out

    def _extend(parts):
        merged = []
        for p in parts:
            if p is None:
                return None
            merged.extend(p)
        return merged

    def _merge_outputs(outs):
        return ASROutput(
            preds=_extend([o.preds for o in outs]),
            logits=_extend([o.logits for o in outs]),
            encoder_logits=None,
            encoder_preds=_extend([o.encoder_preds for o in outs]),
            routes=_extend([o.routes for o in outs]),
            preds_host=_extend([o.preds_host for o in outs]),
        )

    def infer(inp):
        B = _batch_size(inp)
        if exec_batch <= 0 or B <= exec_batch:
            return transcribe(inp)
        outs = []
        for start in range(0, B, exec_batch):
            outs.append(transcribe(_slice_input(inp, start, min(start + exec_batch, B))))
        return _merge_outputs(outs)

    # Prewarm on clones, never by consuming benchmark batches. This keeps every row timed and scored
    # while still compiling the first-call path and every static FRAME_GRID bucket, including ragged B.
    warmup_plan = []
    if batches:
        warmup_plan.extend([batches[0]] * min(warmup_batches, 2))
        seen = set()
        for b in batches:
            inp = prep(b)
            shape_key = (len(b), inp["input_features"].shape[1])
            if frame_grid <= 0 and seen:
                continue
            if shape_key not in seen:
                warmup_plan.append(b)
                seen.add(shape_key)
        for b in warmup_plan:
            infer(prep(b))
            torch.cuda.synchronize()

    # P2.5 double-buffered prep/decode: mel prep for batch i+1 runs on a side CUDA stream while
    # batch i computes; tokenizer decode moves to a worker thread. Pure scheduling — every batch's
    # inputs/outputs are bitwise the batch-serial ones. infer_s is still sync-bounded per batch;
    # e2e_s becomes the honest wall-clock of the whole loop (prep 0 + compute + final decode join).
    from concurrent.futures import ThreadPoolExecutor
    refs, infer_s, audio_s = [], 0.0, 0.0
    route_counts = Counter()
    main_stream = torch.cuda.current_stream()
    prep_stream = torch.cuda.Stream() if overlap else None
    pool = ThreadPoolExecutor(max_workers=2) if overlap else None

    def _prep_work(b):
        # runs on a pool thread: the host-side pinned-buffer fill (the expensive part) AND the
        # GPU mel kernels (side stream) both leave the main thread's critical path.
        # inference_mode is THREAD-LOCAL — re-enter it here or the in-place fill of the pinned
        # buffer (an inference tensor from warmup) raises RuntimeError.
        with torch.inference_mode(), torch.cuda.stream(prep_stream):
            inp = prep(b)
            evt = torch.cuda.Event()
            evt.record(prep_stream)
        return inp, evt

    def prep_async(b):
        return pool.submit(_prep_work, b)

    t_wall0 = time.perf_counter()
    if overlap:
        futures = []
        nxt = prep_async(batches[0]) if batches else None
        for bi, b in enumerate(batches):
            inp, evt = nxt.result()
            nxt = prep_async(batches[bi + 1]) if bi + 1 < len(batches) else None
            main_stream.wait_event(evt)
            for v in inp.values():
                if torch.is_tensor(v):
                    v.record_stream(main_stream)
            evt.synchronize()                       # prep of THIS batch fully done -> not in infer_s
            t0 = time.perf_counter()
            out = infer(inp)                        # MODEL INFERENCE (leaderboard-timed)
            main_stream.synchronize(); infer_s += time.perf_counter() - t0
            if getattr(out, "routes", None):
                route_counts.update(out.routes)
            ids = out.preds_host if getattr(out, "preds_host", None) is not None else out.preds
            futures.append(pool.submit(tok.batch_decode, ids))   # decode on a worker thread
            audio_s += sum(d for _, d, _ in b); refs += [t for _, _, t in b]
        hyps = [h for f in futures for h in f.result()]
        pool.shutdown(wait=True)
    else:
        hyps = []
        for bi, b in enumerate(batches):
            inp = prep(b)                                   # mel extraction EXCLUDED from infer_s
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = infer(inp)                                # MODEL INFERENCE (leaderboard-timed)
            torch.cuda.synchronize(); infer_s += time.perf_counter() - t0
            if getattr(out, "routes", None):
                route_counts.update(out.routes)
            ids = out.preds_host if getattr(out, "preds_host", None) is not None else out.preds
            hyps += tok.batch_decode(ids)                   # decode EXCLUDED from infer_s
            audio_s += sum(d for _, d, _ in b); refs += [t for _, _, t in b]
    e2e_s = time.perf_counter() - t_wall0
    return refs, hyps, audio_s, infer_s, e2e_s, dict(sorted(route_counts.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")            # free label for the row
    ap.add_argument("--levers", default="")                     # comma list; "" = baseline
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "ref"))   # dir with config.json + model.safetensors
    ap.add_argument("--tokenizer", default=None)               # tokenizer.json (default: <model-dir>/tokenizer.json)
    ap.add_argument("--dataset", default="hf-audio/esb-datasets-test-only-sorted")
    ap.add_argument("--config", default="librispeech")
    ap.add_argument("--split", default="test.clean")
    ap.add_argument("--text-key", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true")         # skip the bit-identical probe
    a = ap.parse_args()

    configure_dynamo_limits()

    tok_path = a.tokenizer or os.path.join(a.model_dir, "tokenizer.json")
    model = load_model(a.model_dir, device=DEV)
    fe = MelFeatureExtractor()
    tok = SpeechTokenizer(tok_path)

    rows, tkey = load_data(a.dataset, a.config, a.split, a.text_key, a.max_samples)
    print(f"[bench] variant={a.variant} levers='{a.levers}' frame_grid={FRAME_GRID}")
    print(f"[bench] dataset={a.dataset}/{a.config}:{a.split} text_key={tkey} n={len(rows)} batch={a.batch}")

    # bit-identical probe: capture eager-baseline on the LAST (largest) batch BEFORE applying levers
    probe = None
    if not a.no_probe and len(rows) >= 1:
        pb = rows[-min(a.batch, len(rows)):]
        probe = probe_capture(model, fe, pb)

    transcribe, applied = build(model, a.levers)
    if EXEC_BATCH > 0:
        applied.append(f"exec_batch={EXEC_BATCH}")

    tok_match = n_probe = None; maxd = None
    if probe is not None:
        inp, ref_preds, ref_logits = probe
        tok_match, n_probe, maxd = probe_compare(transcribe, inp, ref_preds, ref_logits)
        print(f"[bench] identity probe vs eager baseline: tokens {tok_match}/{n_probe} identical, max|Δlogit|={maxd}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    refs, hyps, audio_s, infer_s, e2e_s, route_counts = run(
        transcribe, fe, tok, rows, a.batch, frame_grid=FRAME_GRID
    )
    rtfx = round(audio_s / max(e2e_s, 1e-9), 1)
    rtfx_model = round(audio_s / max(infer_s, 1e-9), 1)
    wer_j, wer_k, n_scored = wer_pct(refs, hyps)
    gpu = torch.cuda.get_device_name(0)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    row = {"variant": a.variant, "levers": applied, "gpu": gpu, "dataset": f"{a.config}:{a.split}",
           "split": a.split, "max_samples_arg": a.max_samples, "n": len(refs), "n_scored": n_scored,
           "batch": a.batch,
           "audio_s": round(audio_s, 1), "infer_s": round(infer_s, 2), "e2e_s": round(e2e_s, 2),
           "RTFx": rtfx, "RTFx_model": rtfx_model, "WER_pct": wer_j, "WER_kaldi": wer_k,
           "probe_tok_match": (None if tok_match is None else f"{tok_match}/{n_probe}"),
           "probe_max_dlogit": maxd, "route_counts": route_counts, "peak_vram_gb": peak}
    row = _jsonable(row)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "bench_asr.json")
    allres = json.load(open(path)) if os.path.exists(path) else []
    allres.append(row); json.dump(_jsonable(allres), open(path, "w"), indent=2, allow_nan=False)

    print("\n=== RESULT (in-process, Open ASR Leaderboard timing: mel+model+decode) ===")
    print(f"  GPU {gpu} | {row['dataset']} n={row['n']} scored={row['n_scored']} | batch {a.batch} | levers={applied}")
    print(f"  RTFx={rtfx} (full)  RTFx_model={rtfx_model}  WER={wer_j}% (jiwer)/{wer_k}% (kaldi)  "
          f"VRAM {peak}GB  probe tok={row['probe_tok_match']} max|Δ|={maxd}")
    if route_counts:
        print(f"  routes={route_counts}")
    print("\n| variant | levers | dataset | batch | RTFx | RTFx_model | WER j/k % | probe tok | max|Δlogit| | VRAM | n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in allres:
        print(f"| {r['variant']} | {';'.join(r['levers'])} | {r['dataset']} | {r['batch']} | {r['RTFx']} | "
              f"{r.get('RTFx_model','-')} | {r['WER_pct']}/{r.get('WER_kaldi','-')} | {r.get('probe_tok_match','-')} | "
              f"{r.get('probe_max_dlogit','-')} | {r['peak_vram_gb']}GB | {r['n']} |")


if __name__ == "__main__":
    main()
