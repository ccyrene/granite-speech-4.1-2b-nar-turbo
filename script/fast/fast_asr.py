"""FastGraniteASR — the optimized adaptive CTC-first inference path for Granite Speech 4.1 2B NAR.

This is the shippable config from the A100 optimization study. It is pure ``torch.compile`` (NO TensorRT):
TensorRT was faster per-kernel but bf16-lossy (5.7% WER vs 1.37% baseline), so it is deliberately not used.

What it stacks on top of the bit-identical `models/granite_speech_nar` reimplementation:
  1. GPU-resident feature extraction  — `features.py` moves the waveform to the GPU *before* `pad_sequence`
     (CPU pad_sequence was ~150ms on a fast host; on-GPU it's ~0.03ms). The single biggest e2e win.
  2. torch.compile(encoder)            — branch-free conformer; FRAME_GRID frame-bucketing keeps the
     compiled shapes stable (few recompiles) and is bit-exact.
  3. torch.compile(projector / LLM)    — inductor bf16 == eager numerics (unlike TRT).
  4. text-only LM head                 — runs the tied head only on the text segments (VRAM -16%).
  5. CTC-first adaptive routing        — easy utterances exit on the CTC hypothesis; only hard ones pay
     the full LLM-editor pass. This is the shipped path (≤0.08 WER over the full-editor pass it falls
     back to, large throughput win). See `configs/routing.yaml`.

The compiled stack (1–4) is bit-exact to the eager reimplementation, which is itself head-to-head
identical to the official implementation (LibriSpeech clean 1.38 ≡ 1.38); the adaptive router (5)
adds ≤0.08 WER on every set measured.

`compile_mode="reduce-overhead"` adds CUDA-graph capture (lower single-utterance latency) but needs stable
shapes; the default (None) is shape-flexible and is the config WER-validated at scale.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# repo root on sys.path so `models.granite_speech_nar` imports regardless of CWD
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from models.granite_speech_nar import load_model, MelFeatureExtractor, SpeechTokenizer  # noqa: E402
from models.granite_speech_nar.adaptive import (  # noqa: E402
    length_aware_batches, avg_padding_ratio,
    load_adaptive_config, RoutingConfig, VerifierConfig, EarlyExitConfig,
)
from .longform import chunk_waveform, merge_words  # noqa: E402


def _configure_dynamo_limits(limit: int = 64):
    """Raise graph-cache/recompile limits before any torch.compile call."""
    import torch._dynamo as _dynamo
    cfg = _dynamo.config
    if hasattr(cfg, "cache_size_limit"):
        cfg.cache_size_limit = max(cfg.cache_size_limit, 128)
    if hasattr(cfg, "recompile_limit"):
        cfg.recompile_limit = max(cfg.recompile_limit, limit)
    if hasattr(cfg, "accumulated_recompile_limit"):
        cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, limit)


class FastGraniteASR:
    def __init__(self, model_dir: str, device: str = "cuda", frame_grid: int = 128,
                 compile_mode: str | None = None, text_only_head: bool = True, compile: bool = True,
                 seq_grid: int = 0, routing_config: "str | dict | None" = None):
        """
        model_dir     : dir with config.json + model.safetensors + tokenizer.json
        frame_grid    : pad mel frames to a multiple of this so the compiled encoder sees few shapes
                        (0 = off -> dynamic compile). 128 validated.
        compile_mode  : None (default, shape-flexible) | "reduce-overhead" (CUDA graphs) | "max-autotune"
        text_only_head: run the tied LM head only on text segments (VRAM win, transcript-exact)
        compile       : False -> pure-eager baseline (no torch.compile), for apples-to-apples comparison
        seq_grid      : Phase 7 — bucket the editor's packed [audio;text] length to a multiple of this
                        (0 = off) so the compiled projector/editor see stable shapes (unblocks
                        reduce-overhead / CUDA-graph on the editor).
        routing_config: path/dict for the adaptive-routing thresholds (default: <repo>/configs/routing.yaml).
                        Loaded into self.routing/self.verifier/self.early_exit; missing file or no pyyaml
                        falls back to dataclass defaults. Routing is always enabled here — CTC-first
                        adaptive is the shipped inference path (+45% RTFx, VRAM -29%, ≤0.08 WER over the
                        full-editor pass it falls back to for hard utterances).
        """
        self.device = device
        self.frame_grid = frame_grid if compile else 0   # bucketing only helps the compiled encoder
        self.seq_grid = seq_grid if compile else 0        # packed-length bucketing only helps compiled editor
        self.text_only_head = text_only_head
        self._compiled = bool(compile)
        self.last_routes = None                           # per-sample routes of the last call

        # Adaptive routing config (Phases 2-8). Load once; routing is always enabled here — CTC-first
        # adaptive is the shipped path (the committed configs/routing.yaml may stay enabled:false).
        cfg_src = routing_config if routing_config is not None else os.path.join(_REPO_ROOT, "configs", "routing.yaml")
        try:
            self.routing, self.verifier, self.early_exit = load_adaptive_config(cfg_src)
        except Exception as e:                            # missing file / no pyyaml -> safe defaults
            import warnings
            warnings.warn(f"FastGraniteASR: could not load routing config {cfg_src!r} ({e}); using defaults")
            self.routing, self.verifier, self.early_exit = RoutingConfig(), VerifierConfig(), EarlyExitConfig()
        self.routing.enabled = True                       # adaptive CTC-first routing is the shipped path

        self.model = load_model(model_dir, device=device)
        self.fe = MelFeatureExtractor()
        self.tok = SpeechTokenizer(os.path.join(model_dir, "tokenizer.json"))

        if not compile:
            return                                        # eager baseline

        # many length-buckets over a corpus -> raise the dynamo recompile cache before compiling
        _configure_dynamo_limits()

        m = compile_mode
        # static shapes per bucket when frame_grid>0 (dynamic=False) -> fastest + cudagraph-able
        self.model.encoder = torch.compile(self.model.encoder, dynamic=(self.frame_grid == 0), mode=m)
        self.model.projector = torch.compile(self.model.projector, dynamic=True, mode=m)
        if text_only_head:
            # texthead calls language_model.model(...) -> compile the inner GraniteModel
            self.model.language_model.model = torch.compile(self.model.language_model.model, dynamic=True, mode=m)
        else:
            self.model.language_model = torch.compile(self.model.language_model, dynamic=True, mode=m)

    def _bucket(self, feats: dict) -> dict:
        """Pad mel frames up to a multiple of frame_grid (attention_mask padded with 0 -> real lengths
        preserved, so the extra frames don't change the output). Keeps compiled encoder shapes stable."""
        if self.frame_grid <= 0:
            return feats
        T = feats["input_features"].shape[1]
        fb = ((T + self.frame_grid - 1) // self.frame_grid) * self.frame_grid
        if fb > T:
            feats["input_features"] = F.pad(feats["input_features"], (0, 0, 0, fb - T))
            feats["attention_mask"] = F.pad(feats["attention_mask"], (0, fb - T))
        return feats

    @torch.inference_mode()
    def transcribe(self, waveforms, routing=None, verifier=None, early_exit=None,
                   sample_rate: int = 16000) -> list[str]:
        """CTC-first adaptive transcription (OPTIMIZATION_PLAN Phases 2-8) — the shipped path.

        ``waveforms``: a 1-D tensor/np-array, or a list of them (16 kHz mono). Returns transcripts.
        Easy utterances exit on the CTC hypothesis; only hard ones pay the full LLM-editor pass
        (≤0.08 WER over that full-editor pass, large throughput win). ``routing``/``verifier``/
        ``early_exit`` default to the instance config; pass them to override. Sets ``self.last_routes``
        to the per-sample route list. ``early_exit`` (Phase 5) is eager-only and is ignored when compiled.
        """
        if sample_rate != 16000:
            raise ValueError(f"expected 16 kHz, got {sample_rate} (resampling was removed with torchaudio)")
        if routing is None:
            routing = self.routing
        if verifier is None:
            verifier = self.verifier
        if early_exit is None:
            early_exit = self.early_exit
        if not isinstance(waveforms, (list, tuple)):
            waveforms = [waveforms]
        wavs = [w if torch.is_tensor(w) else torch.as_tensor(w) for w in waveforms]
        feats = self.fe(wavs, device=self.device)            # GPU-resident mel front-end
        feats = self._bucket(feats)
        ee = None if self._compiled else early_exit          # data-dependent break breaks torch.compile
        out = self.model.transcribe_adaptive(**feats, routing=routing, verifier=verifier,
                                             text_only_head=self.text_only_head, early_exit=ee,
                                             seq_grid=self.seq_grid)
        self.last_routes = out.routes
        ids = out.preds_host if getattr(out, "preds_host", None) is not None else out.preds
        return self.tok.batch_decode(ids)

    @torch.inference_mode()
    def transcribe_long(self, waveforms, sample_rate: int = 16000, max_s: float = 30.0,
                        overlap_s: float = 5.0, batch_size: int = 4) -> list[str]:
        """Long-form: split each audio >max_s into overlapping <=max_s windows, batch-transcribe all
        windows, then stitch each audio's windows back (overlap de-dup). Bounds the LLM packed length
        (<=max_s -> well under the 4096 position cap) and keeps shapes uniform for the compiled encoder.
        Inputs <=max_s pass straight through (one window). Returns one transcript per input audio."""
        if sample_rate != 16000:
            raise ValueError(f"expected 16 kHz, got {sample_rate}")
        if not isinstance(waveforms, (list, tuple)):
            waveforms = [waveforms]
        wavs = [(w if torch.is_tensor(w) else torch.as_tensor(w)).reshape(-1).float() for w in waveforms]

        # 1) chunk each audio; remember which audio each window belongs to (preserves order)
        windows, owner = [], []
        for i, w in enumerate(wavs):
            for c in chunk_waveform(w, sample_rate, max_s, overlap_s):
                windows.append(c)
                owner.append(i)

        # 2) transcribe all windows (real lengths -> fe builds the mask; FRAME_GRID buckets the shape).
        # Phase 8: length-aware packing — group windows of similar length so each batch pays minimal
        # padding (the feature extractor pads to the batch max). Original indices are preserved so the
        # per-audio regroup below is unaffected.
        win_words: list[list[str]] = [None] * len(windows)
        win_lengths = [int(w.shape[0]) for w in windows]
        batches = length_aware_batches(win_lengths, batch_size)
        self.last_padding_ratio = avg_padding_ratio(win_lengths, batches)
        for idx_batch in batches:
            texts = self.transcribe([windows[k] for k in idx_batch], sample_rate=sample_rate)
            for k, t in zip(idx_batch, texts):
                win_words[k] = t.split()

        # 3) regroup per audio (in order) and overlap-merge
        out = []
        for i in range(len(wavs)):
            ww = [win_words[k] for k in range(len(windows)) if owner[k] == i]
            out.append(" ".join(merge_words(ww)))
        return out

    @torch.inference_mode()
    def warmup(self, seconds=(5, 10, 20, 30), batch_size: int = 1):
        """Trigger compilation for representative frame-buckets up front.

        Use the deployment batch size here; static compiled encoders guard on both B and the
        FRAME_GRID bucket, so batch=1 warmup does not cover batch=64 serving runs.
        """
        batch_size = max(1, int(batch_size))
        for s in seconds:
            wav = torch.zeros(int(s * 16000))
            self.transcribe([wav] * batch_size)
