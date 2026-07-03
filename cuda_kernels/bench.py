"""Micro-benchmark + correctness helpers for single-kernel validation.

We deliberately measure ONE kernel at a time on representative shapes (never the
whole 2B pipeline — the 6GB card has no headroom), comparing a hand-written CUDA
kernel against (a) the pure-torch reference op for correctness and (b) eager torch
and ``torch.compile`` for speed.
"""
from __future__ import annotations

import torch


def cuda_time_ms(fn, *, warmup: int = 20, iters: int = 100) -> float:
    """Median wall time (ms) of ``fn`` on the current CUDA stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def compare(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Accuracy metrics between a kernel output and the reference."""
    of = out.float()
    rf = ref.float()
    diff = (of - rf).abs()
    denom = rf.abs().clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "exact": bool(torch.equal(out, ref)),
        "dtype": str(out.dtype),
        "shape": tuple(out.shape),
    }


def report(name: str, acc: dict, t_kernel: float, t_eager: float, t_compile: float | None = None):
    tag = "EXACT max|Δ|=0" if acc["exact"] else f"max|Δ|={acc['max_abs']:.3e} rel={acc['max_rel']:.2e}"
    line = (f"  {name:<28} {acc['shape']!s:<18} {acc['dtype']:<14} {tag}\n"
            f"      kernel {t_kernel*1000:8.2f}us   eager {t_eager*1000:8.2f}us   "
            f"speedup {t_eager/t_kernel:5.2f}x")
    if t_compile is not None:
        line += f"   compile {t_compile*1000:8.2f}us ({t_compile/t_kernel:5.2f}x)"
    print(line)
    return acc["exact"]
