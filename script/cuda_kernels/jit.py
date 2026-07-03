"""Hand-written CUDA kernel JIT harness (NVRTC + CUDA driver API).

Why NVRTC and not ``torch.utils.cpp_extension`` / offline ``nvcc``:
the only CUDA toolkit on this box is 12.0, whose front-end (``cicc``/``cudafe++``)
cannot parse gcc-13 / glibc-2.39 system headers (``_Float32`` undefined, libstdc++13
builtins). NVRTC 12.8 (matching torch's CUDA runtime) compiles device code with its
own self-contained headers, so there is no system-header conflict. Kernels are real
hand-written CUDA C++ compiled to PTX at runtime, loaded via the driver API, and
launched on torch's *current* stream so they interleave correctly with torch ops.

Usage:
    from cuda_kernels.jit import Kernel
    k = Kernel(SRC, "my_kernel")           # compiles + caches
    k.launch(grid, block, args, shared=0)  # args: list built with arg_* helpers
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from typing import Sequence

import torch
from cuda.bindings import driver as _drv
from cuda.bindings import nvrtc as _nvrtc

_PTX_CACHE_DIR = os.environ.get(
    "GRANITE_PTX_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "granite_ptx"),
)
os.makedirs(_PTX_CACHE_DIR, exist_ok=True)


def _module_dir(name: str) -> str | None:
    """Package dir; namespace packages (e.g. the cu13 nvidia wheels) have __file__=None,
    so fall back to __path__."""
    try:
        import importlib
        m = importlib.import_module(name)
    except Exception:
        return None
    f = getattr(m, "__file__", None)
    if f:
        return os.path.dirname(f)
    try:
        return next(iter(m.__path__))
    except Exception:
        return None


def _cuda_include_dirs() -> list[str]:
    """Dirs with cuda_bf16.h and crt/mma.h from the pip CUDA wheels.
    cu12 layout: nvidia/cuda_runtime + nvidia/cuda_nvcc; cu13 layout: one nvidia/cu13 dir."""
    dirs = []
    for name in ("nvidia.cu13", "nvidia.cuda_runtime"):
        d = _module_dir(name)
        if d and os.path.exists(os.path.join(d, "include", "cuda_bf16.h")):
            dirs.append(os.path.join(d, "include"))
            break
    if not dirs:
        raise RuntimeError("cuda_bf16.h not found (install nvidia-cuda-runtime-cu12/-cu13)")
    for name in ("nvidia.cu13", "nvidia.cuda_nvcc"):
        d = _module_dir(name)
        inc = d and os.path.join(d, "include")
        if inc and os.path.exists(os.path.join(inc, "crt", "mma.h")):
            if inc not in dirs:
                dirs.append(inc)
            break
    return dirs


_CUDA_INCS = _cuda_include_dirs()


# --------------------------------------------------------------------------- #
# error checking
# --------------------------------------------------------------------------- #
def _ck(*args):
    """Unwrap a cuda-python ``(result, *outs)`` return; raise on error."""
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]
    err, rest = args[0], args[1:]
    if isinstance(err, _nvrtc.nvrtcResult):
        if err != _nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err}")
    elif isinstance(err, _drv.CUresult):
        if err != _drv.CUresult.CUDA_SUCCESS:
            _, name = _drv.cuGetErrorString(err)
            raise RuntimeError(f"CUDA driver error {int(err)}: {name.decode() if isinstance(name, bytes) else name}")
    else:
        raise RuntimeError(f"unexpected status type {type(err)}: {err}")
    return rest[0] if len(rest) == 1 else rest


def _arch_flag() -> bytes:
    major, minor = torch.cuda.get_device_capability()
    return f"--gpu-architecture=compute_{major}{minor}".encode()


# --------------------------------------------------------------------------- #
# kernel argument helpers  ->  ctypes objects (kept alive by the caller list)
# --------------------------------------------------------------------------- #
def arg_ptr(t: torch.Tensor):
    """Device pointer of a CUDA tensor."""
    assert t.is_cuda, "tensor must be on CUDA"
    return ctypes.c_void_p(t.data_ptr())

def arg_i32(x: int):
    return ctypes.c_int32(int(x))

def arg_i64(x: int):
    return ctypes.c_int64(int(x))

def arg_f32(x: float):
    return ctypes.c_float(float(x))


def _build_param_array(args: Sequence):
    """Pack ctypes args into a void** parameter array for cuLaunchKernel."""
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    return arr


# --------------------------------------------------------------------------- #
# compile + launch
# --------------------------------------------------------------------------- #
_PROG_CACHE: dict[str, "Kernel"] = {}


def _compile_ptx(src: str, name: str, extra_opts: tuple[bytes, ...]) -> bytes:
    arch = _arch_flag()
    key = hashlib.sha256(
        (src + "|" + arch.decode() + "|" + "|".join(o.decode() for o in extra_opts)).encode()
    ).hexdigest()[:24]
    cache_path = os.path.join(_PTX_CACHE_DIR, f"{name}_{key}.ptx")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    prog = _ck(_nvrtc.nvrtcCreateProgram(src.encode(), (name + ".cu").encode(), 0, [], []))
    opts = [arch, b"--std=c++17", b"-default-device"] + [f"-I{d}".encode() for d in _CUDA_INCS] + list(extra_opts)
    res = _nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log_size = _ck(_nvrtc.nvrtcGetProgramLogSize(prog))
    if log_size > 1:
        log = b" " * log_size
        _nvrtc.nvrtcGetProgramLog(prog, log)
        msg = log.decode(errors="replace").strip()
        if res[0] != _nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC compile failed for {name}:\n{msg}")
        elif msg:
            print(f"[nvrtc {name}] {msg}")
    _ck(*res)
    ptx_size = _ck(_nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    _ck(_nvrtc.nvrtcGetPTX(prog, ptx))
    _nvrtc.nvrtcDestroyProgram(prog)
    with open(cache_path, "wb") as f:
        f.write(ptx)
    return ptx


class Kernel:
    """A single compiled CUDA kernel, launchable on torch's current stream."""

    def __init__(self, src: str, name: str, extra_opts: tuple[bytes, ...] = ()):
        self.src = src
        self.name = name
        torch.cuda.init()
        ptx = _compile_ptx(src, name, extra_opts)
        self.module = _ck(_drv.cuModuleLoadData(ptx))
        self.func = _ck(_drv.cuModuleGetFunction(self.module, name.encode()))

    def launch(self, grid, block, args: Sequence, shared: int = 0, stream=None):
        if isinstance(grid, int):
            grid = (grid, 1, 1)
        if isinstance(block, int):
            block = (block, 1, 1)
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        params = _build_param_array(args)
        _ck(_drv.cuLaunchKernel(
            self.func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared, stream, ctypes.addressof(params), 0,
        ))


def get_kernel(src: str, name: str, extra_opts: tuple[bytes, ...] = ()) -> Kernel:
    """Compile-or-fetch a cached kernel keyed by (src, name, opts)."""
    key = name + "|" + hashlib.sha256(src.encode()).hexdigest()[:16] + "|" + "|".join(o.decode() for o in extra_opts)
    k = _PROG_CACHE.get(key)
    if k is None:
        k = Kernel(src, name, extra_opts)
        _PROG_CACHE[key] = k
    return k
