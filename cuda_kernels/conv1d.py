"""Fused depthwise Conv1d (k=15) + bias + SiLU  (conformer conv module, post BN-fold).

After fold_bn, the conv module's depthwise conv carries a bias and BatchNorm becomes
Identity, so the block is: silu(depthwise_conv(x) + bias). Reference:
    silu(F.conv1d(F.pad(x,(7,7)), w(C,1,15), bias, groups=C))
out[b,c,t] = silu( bias[c] + sum_{j<15} w[c,j] * x[b,c, t-7+j] )  (zero outside [0,T)).
fp32 accumulation; threads ordered so a warp shares (b,c) -> weights broadcast from
cache, consecutive t -> coalesced x. Bar: WER/argmax-safe (the fold itself is not
bit-identical); compared vs eager Conv1d+SiLU and vs fp64 golden.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .jit import get_kernel, arg_ptr, arg_i32

_SRC = r"""
#include <cuda_bf16.h>

extern "C" __global__ void dwconv1d_silu_bf16(
        const __nv_bfloat16* __restrict__ x,    // (B, C, T)
        const __nv_bfloat16* __restrict__ w,    // (C, 1, K)
        const __nv_bfloat16* __restrict__ bias, // (C,)
        __nv_bfloat16* __restrict__ y,          // (B, C, T)
        int B, int C, int T, int K, int pad) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = (long long)B * C * T;
    if (i >= tot) return;
    int t = i % T; long long bc = i / T;
    int c = bc % C; int b = bc / C;
    long long xbase = ((long long)b * C + c) * T;
    long long wbase = (long long)c * K;
    float acc = __bfloat162float(bias[c]);
    for (int j = 0; j < K; ++j) {
        int ti = t - pad + j;
        if (ti >= 0 && ti < T)
            acc += __bfloat162float(w[wbase + j]) * __bfloat162float(x[xbase + ti]);
    }
    acc = acc / (1.0f + expf(-acc));   // silu, precise
    y[i] = __float2bfloat16_rn(acc);
}
"""

_SRC_GLU = r"""
#include <cuda_bf16.h>

// GLU folded into the depthwise conv: x2 is the up_conv output (B, 2C, T), a = x2[:, :C],
// gate = x2[:, C:]; out = silu(bias + conv_k(a * sigmoid(gate))). The (B, C, T) GLU tensor
// never exists (saves one full write+read per layer). The glu product is rounded to bf16
// before accumulation to match what the eager chain would have materialized.
extern "C" __global__ void dwconv1d_silu_glu_bf16(
        const __nv_bfloat16* __restrict__ x2,   // (B, 2C, T)
        const __nv_bfloat16* __restrict__ w,    // (C, 1, K)
        const __nv_bfloat16* __restrict__ bias, // (C,)
        __nv_bfloat16* __restrict__ y,          // (B, C, T)
        int B, int C, int T, int K, int pad) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = (long long)B * C * T;
    if (i >= tot) return;
    int t = i % T; long long bc = i / T;
    int c = bc % C; int b = bc / C;
    long long abase = ((long long)b * 2 * C + c) * T;
    long long gbase = abase + (long long)C * T;
    long long wbase = (long long)c * K;
    float acc = __bfloat162float(bias[c]);
    for (int j = 0; j < K; ++j) {
        int ti = t - pad + j;
        if (ti >= 0 && ti < T) {
            float av = __bfloat162float(x2[abase + ti]);
            float gv = __bfloat162float(x2[gbase + ti]);
            float glu = av / (1.0f + expf(-gv));
            glu = __bfloat162float(__float2bfloat16_rn(glu));
            acc += __bfloat162float(w[wbase + j]) * glu;
        }
    }
    acc = acc / (1.0f + expf(-acc));   // silu, precise
    y[i] = __float2bfloat16_rn(acc);
}
"""

_BLOCK = 256


def dwconv1d_silu_glu(x2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """x2 (B,2C,T) = up_conv output; weight (C,1,K), bias (C,). Returns
    silu(conv(glu(x2)) + bias) with GLU computed inline (no (B,C,T) intermediate)."""
    B, C2, T = x2.shape
    C = C2 // 2
    K = weight.shape[-1]
    pad = K // 2
    xc = x2.contiguous(); wc = weight.contiguous(); bc = bias.contiguous()
    y = torch.empty((B, C, T), device=x2.device, dtype=x2.dtype)
    n = B * C * T
    grid = ((n + _BLOCK - 1) // _BLOCK, 1, 1)
    get_kernel(_SRC_GLU, "dwconv1d_silu_glu_bf16").launch(grid, _BLOCK,
        [arg_ptr(xc), arg_ptr(wc), arg_ptr(bc), arg_ptr(y),
         arg_i32(B), arg_i32(C), arg_i32(T), arg_i32(K), arg_i32(pad)])
    return y


def dwconv1d_silu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """x (B,C,T), weight (C,1,K), bias (C,). Symmetric pad K//2. Returns silu(conv+bias)."""
    B, C, T = x.shape
    K = weight.shape[-1]
    pad = K // 2
    xc = x.contiguous(); wc = weight.contiguous(); bc = bias.contiguous()
    y = torch.empty_like(xc)
    n = B * C * T
    grid = ((n + _BLOCK - 1) // _BLOCK, 1, 1)
    get_kernel(_SRC, "dwconv1d_silu_bf16").launch(grid, _BLOCK,
        [arg_ptr(xc), arg_ptr(wc), arg_ptr(bc), arg_ptr(y),
         arg_i32(B), arg_i32(C), arg_i32(T), arg_i32(K), arg_i32(pad)])
    return y


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .bench import cuda_time_ms, compare, report

    torch.manual_seed(0)
    C, K = 2048, 15
    print("== Depthwise Conv1d k=15 + bias + SiLU (conformer, inner=2048) ==")
    ok_all = True
    conv = torch.nn.Conv1d(C, C, K, groups=C, bias=True).cuda().bfloat16()
    conv.weight.data.normal_(0, 0.1); conv.bias.data.normal_(0, 0.05)

    def ref(x):
        return F.silu(conv(F.pad(x, (K // 2, K // 2))))
    cref = torch.compile(ref)

    for T in (261, 843):
        x = torch.randn(1, C, T, device="cuda", dtype=torch.bfloat16)
        r = ref(x); o = dwconv1d_silu(x, conv.weight, conv.bias); acc = compare(o, r)
        # fp64 golden
        wg = conv.weight.double(); bg = conv.bias.double()
        rg = F.silu(F.conv1d(F.pad(x.double(), (K // 2, K // 2)), wg, bg, groups=C)).to(torch.bfloat16)
        dk = (o - rg).abs().max().item(); de = (r - rg).abs().max().item()
        _ = cref(x)
        t_k = cuda_time_ms(lambda: dwconv1d_silu(x, conv.weight, conv.bias))
        t_e = cuda_time_ms(lambda: ref(x)); t_c = cuda_time_ms(lambda: cref(x))
        report(f"dwconv T={T}", acc, t_k, t_e, t_c)
        print(f"      vs fp64-golden: kernel {dk:.3e}  eager {de:.3e}  -> {'kernel >= eager faithful' if dk <= de else 'check'}")
        ok_all = ok_all and dk <= max(de * 1.5, 1.6e-2)
    print("RESULT:", "fused conv+bias+silu, kernel as-faithful-as-eager PASS" if ok_all else "CHECK")
