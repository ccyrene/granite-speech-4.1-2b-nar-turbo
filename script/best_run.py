#!/usr/bin/env python
"""Run the BEST measured bench config (see best/__init__.py) via script/bench_asr.py.

    python script/best_run.py [--model-dir ref] [--split test.clean] [--max-samples 500]
                       [--batch 128] [--exec-batch 48] [--variant best] [--probe]

Defaults reproduce the session-C record run (3934.7 RTFx e2e / 4025.7 model on H100 SXM):
logical b128, EXEC_BATCH=48, FRAME_GRID=128, encoder max-autotune-no-cudagraphs, all
champion levers + the fused dwconv custom kernel. Extra args after `--` are passed through
to script/bench_asr.py unchanged.
"""
import argparse
import os
import subprocess
import sys

SCRIPT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT)
sys.path.insert(0, SCRIPT)
from best import BEST_LEVERS, BEST_BATCH, apply_best_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "ref"))
    ap.add_argument("--config", default="librispeech")
    ap.add_argument("--split", default="test.clean")
    ap.add_argument("--batch", type=int, default=BEST_BATCH)
    ap.add_argument("--exec-batch", type=int, default=48)
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--variant", default="best")
    ap.add_argument("--probe", action="store_true",
                    help="keep the eager-baseline probe (default: --no-probe)")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    env = apply_best_env()
    env["EXEC_BATCH"] = str(args.exec_batch)

    cmd = [sys.executable, os.path.join(SCRIPT, "bench_asr.py"),
           "--variant", args.variant, "--levers", BEST_LEVERS,
           "--model-dir", args.model_dir, "--config", args.config, "--split", args.split,
           "--batch", str(args.batch), "--max-samples", str(args.max_samples)]
    if not args.probe:
        cmd.append("--no-probe")
    cmd += extra

    print("[best] env:", {k: env[k] for k in ("FRAME_GRID", "EXEC_BATCH", "ENC_COMPILE_MODE")})
    print("[best] cmd:", " ".join(cmd))
    return subprocess.call(cmd, env=env, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
