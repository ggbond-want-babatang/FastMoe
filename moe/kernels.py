"""Triton 3.1 compatible persistent grouped GEMM and deterministic combine."""
import torch
import triton
import triton.language as tl
from .reference import route, validate, pack_routes, swiglu


@triton.jit
def _grouped(A, W, C, OFF, N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
             SWIGLU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr, WORKERS: tl.constexpr):
    tile = tl.program_id(0)
    # Offsets may be int64; keep the loop-carried tile boundary type stable.
    end = tl.full((), 0, tl.int64)
    # Every worker walks the compact expert tile space on device.
    for expert in range(E):
        start = tl.load(OFF + expert)
        stop = tl.load(OFF + expert + 1)
        m = stop - start
        nt = tl.cdiv(N, BN)
        total = tl.cdiv(m, BM) * nt
        while (tile >= end) & (tile < end + total):
            local = tile - end
            rows = (local // nt) * BM + tl.arange(0, BM)
            cols = (local % nt) * BN + tl.arange(0, BN)
            kk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            if SWIGLU:
                acc_up = tl.zeros((BM, BN), tl.float32)
                width = 2 * N
            else:
                width = N
            for block in range(tl.cdiv(K, BK)):
                ks = block * BK + kk
                aa = tl.load(A + (start + rows[:, None])*K + ks[None, :],
                             (rows[:, None] < m) & (ks[None, :] < K), 0)
                ww = tl.load(W + expert*K*width + ks[:, None]*width + cols[None, :],
                             (ks[:, None] < K) & (cols[None, :] < N), 0)
                acc += tl.dot(aa, ww)
                if SWIGLU:
                    wu = tl.load(W + expert*K*width + ks[:, None]*width + N + cols[None, :],
                                 (ks[:, None] < K) & (cols[None, :] < N), 0)
                    acc_up += tl.dot(aa, wu)
            if SWIGLU:
                # Match reference's projection rounding before FP32 activation.
                gate = acc.to(A.dtype.element_ty).to(tl.float32)
                up = acc_up.to(A.dtype.element_ty).to(tl.float32)
                acc = gate * tl.sigmoid(gate) * up
            tl.store(C + (start + rows[:, None])*N + cols[None, :], acc,
                     (rows[:, None] < m) & (cols[None, :] < N))
            tile += WORKERS
        end += total


@triton.jit
def _activation(GU, OUT, F: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    ix = tl.program_id(0)*B + tl.arange(0, B)
    row, col = ix // F, ix % F
    gate = tl.load(GU + row*2*F + col, ix < TOTAL, 0).to(tl.float32)
    up = tl.load(GU + row*2*F + F + col, ix < TOTAL, 0).to(tl.float32)
    tl.store(OUT + ix, gate * tl.sigmoid(gate) * up, ix < TOTAL)


@triton.jit
def _combine(Y, INV, P, OUT, H: tl.constexpr, TOP_K: tl.constexpr,
             TOTAL: tl.constexpr, B: tl.constexpr):
    ix = tl.program_id(0)*B + tl.arange(0, B)
    token, col = ix // H, ix % H
    acc = tl.zeros((B,), tl.float32)
    for slot in tl.static_range(TOP_K):
        row = tl.load(INV + token*TOP_K + slot, ix < TOTAL, 0)
        p = tl.load(P + token*TOP_K + slot, ix < TOTAL, 0)
        value = tl.load(Y + row*H + col, ix < TOTAL, 0).to(tl.float32)
        acc += value * p
    tl.store(OUT + ix, acc, ix < TOTAL)


def grouped_mm(a, weights, offsets, fused_swiglu=False, block_m=32, block_n=64):
    """A[R,K], W[E,K,N], GPU offsets[E+1]. Contiguous FP16/BF16 only.

    Offsets must be nondecreasing, begin with zero and end at R. Hot-path
    validation intentionally avoids GPU synchronization; caller owns metadata.
    """
    if a.ndim != 2 or weights.ndim != 3 or offsets.shape != (weights.shape[0]+1,):
        raise ValueError("invalid grouped GEMM shape")
    if a.dtype not in (torch.float16, torch.bfloat16) or weights.dtype != a.dtype:
        raise ValueError("grouped GEMM requires matching FP16/BF16")
    if not a.is_cuda or weights.device != a.device or offsets.device != a.device:
        raise ValueError("all grouped GEMM tensors must share a CUDA device")
    if not all(t.is_contiguous() for t in (a, weights, offsets)):
        raise ValueError("grouped GEMM tensors must be contiguous")
    if offsets.dtype not in (torch.int32, torch.int64) or a.shape[1] != weights.shape[1]:
        raise ValueError("invalid offsets dtype or contraction dimension")
    r, k = a.shape
    e, _, width = weights.shape
    if fused_swiglu and width % 2:
        raise ValueError("fused gate/up width must be even")
    n = width//2 if fused_swiglu else width
    if min(k, n, e) <= 0 or block_m not in (16, 32, 64) or block_n not in (32, 64, 128):
        raise ValueError("unsupported dimensions or tile size")
    out = torch.empty((r, n), device=a.device, dtype=a.dtype)
    if r:
        with torch.cuda.device(a.device):
            sms = torch.cuda.get_device_properties(a.device).multi_processor_count
            upper = (triton.cdiv(r, block_m) + e - 1) * triton.cdiv(n, block_n)
            workers = min(2*sms, upper)
            _grouped[(workers,)](a, weights, out, offsets, n, k, e, fused_swiglu,
                                block_m, block_n, 32, workers, num_warps=4, num_stages=2)
    return out


def fused_activation(gu):
    rows, width = gu.shape
    out = torch.empty((rows, width//2), dtype=gu.dtype, device=gu.device)
    if out.numel():
        _activation[(triton.cdiv(out.numel(), 256),)](gu, out, width//2, out.numel(), 256)
    return out


def combine(y, inverse, probs, shape):
    out = torch.empty(shape, dtype=y.dtype, device=y.device)
    if out.numel():
        _combine[(triton.cdiv(out.numel(), 256),)](y, inverse, probs.contiguous(), out,
                                              shape[1], probs.shape[1], out.numel(), 256)
    return out


@torch.no_grad()
def moe_triton(x, w, top_k=2, routing=None, fuse_projection=True):
    validate(x, w)
    ids, probs = route(x, w.router, top_k) if routing is None else routing
    a, offsets, inverse = pack_routes(x, ids, w.gate_up.shape[0])
    if fuse_projection:
        hidden = grouped_mm(a, w.gate_up, offsets, fused_swiglu=True)
    else:
        hidden = fused_activation(grouped_mm(a, w.gate_up, offsets))
    values = grouped_mm(hidden, w.down, offsets)
    return combine(values, inverse, probs, x.shape)
