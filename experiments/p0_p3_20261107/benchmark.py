"""Reproducible CUDA-event and wall-clock inference benchmarks, no CUDA graphs."""
import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from moe.reference import make_weights, moe_reference, pack_routes, synthetic_routing, swiglu
from moe.kernels import moe_triton, grouped_mm


def quantile(values, q):
    values = sorted(values)
    x = q*(len(values)-1)
    lo = int(x)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


@torch.no_grad()
def measure(fn, warmup, samples, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    cuda_us, wall_us = [], []
    for _ in range(samples):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tick = time.perf_counter()
        start.record()
        for _ in range(repeats):
            fn()
        stop.record()
        stop.synchronize()
        cuda_us.append(start.elapsed_time(stop)*1000/repeats)
        wall_us.append((time.perf_counter()-tick)*1e6/repeats)
    gc.collect()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return dict(cuda_median_us=statistics.median(cuda_us), cuda_p10_us=quantile(cuda_us, .1),
                cuda_p90_us=quantile(cuda_us, .9), wall_median_us=statistics.median(wall_us),
                peak_allocated_bytes=peak, incremental_peak_bytes=peak-allocated,
                cuda_samples_us=cuda_us, wall_samples_us=wall_us)


def packed_reference(a, weights, counts, activation=False):
    values = []
    offset = 0
    for expert, count in enumerate(counts):
        values.append(a[offset:offset+count] @ weights[expert])
        offset += count
    out = torch.cat(values, dim=0)
    return swiglu(out) if activation else out


def assert_correct(actual, expected, dtype):
    atol, rtol = (.004, .02) if dtype == torch.float16 else (.035, .06)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "standard", "full"), default="standard")
    parser.add_argument("--tokens", type=int, nargs="+", default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to produce CPU benchmark numbers")
    if min(args.hidden, args.ffn, args.warmup, args.repeats) < 1 or args.samples < 3:
        parser.error("positive dimensions/warmup/repeats and >=3 samples required")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tokens = args.tokens or ([1, 128] if args.profile == "quick" else [1, 8, 32, 128, 512, 2048])
    if min(tokens) < 1:
        parser.error("benchmark tokens must be positive")
    ek = [(8, 2)]
    if args.profile == "full":
        ek = [(e, k) for e in (4, 8, 16) for k in (1, 2, 4)]
    configs = [(t, e, k) for e, k in ek for t in tokens]
    if args.profile == "standard":
        configs += [(128, 4, 1), (128, 16, 4)]
    records = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(dict(arguments=vars(args), gpu=torch.cuda.get_device_name(0),
                                         torch=torch.__version__, records=records), indent=2))
        if records:
            fields = [key for key in records[0] if not key.endswith("samples_us")]
            with output.with_suffix(".csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

    def run_group(meta, methods, flops, expected):
        base = None
        for name, fn in methods:
            assert_correct(fn(), expected, dtype)  # Compilation and validation excluded from timing.
            row = dict(**meta, implementation=name, **measure(fn, args.warmup, args.samples, args.repeats))
            if base is None:
                base = row["cuda_median_us"]
            row.update(speedup_vs_torch=base/row["cuda_median_us"],
                       tokens_per_second=meta["tokens"]*1e6/row["cuda_median_us"],
                       effective_tflops=flops/(row["cuda_median_us"]*1e6))
            records.append(row)
            print(f'{meta["scope"]:18s} T={meta["tokens"]:4d} E={meta["experts"]:2d} '
                  f'K={meta["top_k"]} {meta["distribution"]:7s} {name:16s} '
                  f'{row["cuda_median_us"]:9.2f} us {row["speedup_vs_torch"]:5.2f}x', flush=True)
        save()  # Preserve completed measurements if a later case fails.

    for t, e, k in configs:
        x = torch.randn(t, args.hidden, device="cuda", dtype=dtype)
        w = make_weights(args.hidden, args.ffn, e, dtype=dtype)
        meta = dict(tokens=t, experts=e, top_k=k, hidden=args.hidden, ffn=args.ffn,
                    dtype=args.dtype, scope="end_to_end", distribution="router")
        flops = 6*t*k*args.hidden*args.ffn
        run_group(meta, [("torch_loop", lambda: moe_reference(x, w, k)),
                         ("triton_unfused", lambda: moe_triton(x, w, k, fuse_projection=False)),
                         ("triton_fused", lambda: moe_triton(x, w, k))],
                  flops, moe_reference(x, w, k))
        for dist in ("uniform", "random", "skewed"):
            routing = synthetic_routing(t, e, k, dist)
            meta.update(scope="routed_moe", distribution=dist)
            run_group(meta, [("torch_loop", lambda: moe_reference(x, w, k, routing)),
                             ("triton_fused", lambda: moe_triton(x, w, k, routing))],
                      flops, moe_reference(x, w, k, routing))
            a, offsets, inverse = pack_routes(x, routing[0], e)
            counts = (offsets[1:]-offsets[:-1]).tolist()  # Outside timing, reference only.
            meta.update(scope="grouped_gemm")
            expected = packed_reference(a, w.gate_up, counts)
            run_group(meta, [("torch_cublas", lambda: packed_reference(a, w.gate_up, counts)),
                             ("triton_grouped", lambda: grouped_mm(a, w.gate_up, offsets))],
                      4*t*k*args.hidden*args.ffn, expected)
            meta.update(scope="gate_up_swiglu")
            run_group(meta, [("torch_cublas", lambda: packed_reference(a, w.gate_up, counts, True)),
                             ("triton_fused", lambda: grouped_mm(a, w.gate_up, offsets, True))],
                      4*t*k*args.hidden*args.ffn, swiglu(expected))
            if args.tune and t == 512 and e == 8 and k == 2 and dist == "random":
                meta.update(scope="tile_sweep")
                methods = [("torch_cublas", lambda: packed_reference(a, w.gate_up, counts, True))]
                for bm in (16, 32, 64):
                    for bn in (32, 64, 128):
                        methods.append((f"BM{bm}_BN{bn}", lambda bm=bm, bn=bn:
                                        grouped_mm(a, w.gate_up, offsets, True, bm, bn)))
                run_group(meta, methods, 4*t*k*args.hidden*args.ffn, swiglu(expected))
        del x, w, a, offsets, inverse, expected
    print(f"Saved {len(records)} measurements to {output}")


if __name__ == "__main__":
    main()
