"""`best/` — the fastest measured Granite Speech 4.1 2B NAR bench config (H100 session C).

Best e2e: **3934.7 RTFx** (model-only 4025.7) on H100 SXM, LibriSpeech test.clean subset-500,
WER 1.27/1.09 (variant `h100c_conv_b128ex48`; H200 measures 3951.5 with the same config).

The config = session-B champion + session-C fused conv kernel, executed as logical batch 128
split into EXEC_BATCH=48 chunks (avoids the b>=64 max-autotune illegal-address shapes and
trims each chunk to its own length bucket):

- levers: compile-enc/proj/llm, texthead, adaptive (CTC-first routing), flexattn (editor),
  encattn (fused SDPA encoder), encdense (single-graph dense BPE head), freeze (inductor),
  convkernel (granite::dwconv_silu — fused depthwise conv+bias+SiLU after BN fold)
- env: FRAME_GRID=128, EXEC_BATCH=48, ENC_COMPILE_MODE=max-autotune-no-cudagraphs

WER-gated deviation class (encattn + BN fold are not bit-exact; transcripts verified exact on
reference goldens; full-set gates PASS: clean 1.42/1.19, other 2.81/2.52).

Usage:
    python script/best_run.py                      # subset-500 bench on ref/ (defaults = the record run)
    python script/best_run.py --max-samples 64     # quick smoke
    python script/best_run.py --split test.other --max-samples 0   # full set
"""

BEST_LEVERS = ("compile-enc,compile-proj,compile-llm,texthead,"
               "adaptive,flexattn,encattn,encdense,freeze,convkernel")

BEST_ENV = {
    "FRAME_GRID": "128",
    "EXEC_BATCH": "48",
    "ENC_COMPILE_MODE": "max-autotune-no-cudagraphs",
}

BEST_BATCH = 128


def apply_best_env(env=None):
    """Overlay the best-config env vars onto `env` (defaults to a copy of os.environ)."""
    import os
    e = dict(os.environ if env is None else env)
    e.update(BEST_ENV)
    return e
