"""Demo: transcribe a wav with the FastGraniteASR path + report warm latency / RTFx.

  python script/fast_demo.py [--model-dir DIR] [--wav PATH] [--mode reduce-overhead] [--lossless]

Defaults to the adaptive CTC-first winner (+45% RTFx / VRAM -29%, near-lossless).
Pass --lossless to fall back to the strictly bit-exact `transcribe` path.
Defaults to the bundled sample (ref/10226_10111_000000.wav). model-dir defaults to ref/ (or set
MODEL_DIR, e.g. an HF snapshot dir with config.json + model.safetensors + tokenizer.json).
"""
import argparse
import os
import sys
import time

import torch

SCRIPT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT)
sys.path.insert(0, SCRIPT)
from fast import FastGraniteASR  # noqa: E402


def load_wav(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return torch.from_numpy(a), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", os.path.join(ROOT, "ref")))
    ap.add_argument("--wav", default=os.path.join(ROOT, "ref", "10226_10111_000000.wav"))
    ap.add_argument("--mode", default=None, help="None | reduce-overhead | max-autotune")
    ap.add_argument("--lossless", action="store_true",
                    help="use the strictly bit-exact transcribe path (default: adaptive CTC-first)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    adaptive = not args.lossless
    asr = FastGraniteASR(args.model_dir, device=dev, compile_mode=args.mode, adaptive=adaptive)
    fn = asr.transcribe if args.lossless else asr.transcribe_adaptive
    path = "lossless" if args.lossless else "adaptive"
    wav, sr = load_wav(args.wav)
    secs = wav.shape[-1] / sr
    print(f"device={dev} mode={args.mode} path={path} | audio={secs:.2f}s")

    # warm (compiles the bucket for this length)
    txt = fn(wav, sample_rate=sr)[0]
    for _ in range(3):
        fn(wav, sample_rate=sr)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        fn(wav, sample_rate=sr)
    if dev == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1e3
    print(f"\nwarm e2e: {ms:.2f} ms  | RTFx (single utterance): {secs / (ms / 1e3):.1f}")
    if dev == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    if adaptive:
        print(f"routes: {asr.last_routes}")   # ['ctc_fast' | 'local' | 'full'] per input
    print(f"\nTRANSCRIPT:\n{txt}")


if __name__ == "__main__":
    main()
