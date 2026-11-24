"""P0: independent uninstrumented timing, stage events, Kineto, and NVTX."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import torch
from benchmark import measure, assert_correct
from moe.reference import make_weights, moe_reference
from moe.kernels import moe_triton
from moe.instrumentation import Recorder, stage


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("torch", "nvtx", "timing"), default="torch")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 2048])
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    root = Path("results/p0")
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261107)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    w = make_weights(256, 512, 8)
    cases = []
    for t in args.tokens:
        x = torch.randn(t, 256, device="cuda", dtype=torch.float16)
        assert_correct(moe_triton(x, w), moe_reference(x, w), torch.float16)
        for name, fn in (("baseline", lambda x=x: moe_reference(x, w)),
                         ("triton", lambda x=x: moe_triton(x, w))):
            for _ in range(10):
                fn()
            cases.append((t, name, fn))
    torch.cuda.synchronize()
    if args.mode == "nvtx":
        torch.cuda.cudart().cudaProfilerStart()
        recorder = Recorder(events=False, nvtx=True)
        with recorder.activate():
            for t, name, fn in cases:
                for _ in range(args.iterations):
                    with stage(f"case/T{t}/{name}"):
                        fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        (root / "nvtx_run.json").write_text(json.dumps(dict(mode=args.mode, iterations=args.iterations,
            tokens=args.tokens, captured_regions=len(recorder.records)), indent=2))
        print("NVTX capture completed", flush=True)
        return
    results = []
    for t, name, fn in cases:
        # These timings run before any profiling context; do not conflate with traces.
        bare = measure(fn, warmup=10, samples=15, repeats=3)
        recorder = Recorder(events=True, nvtx=False)
        with recorder.activate():
            for _ in range(args.iterations):
                fn()
        events = recorder.resolved()
        grouped = defaultdict(list)
        cpu = defaultdict(list)
        for event in events:
            grouped[event["stage"]].append(event["cuda_interval_us"])
            cpu[event["stage"]].append(event["cpu_region_us"])
        phases = {key:dict(median_cuda_interval_us=statistics.median(vals),
                          median_cpu_region_us=statistics.median(cpu[key]), samples_us=vals)
                  for key, vals in grouped.items()}
        row = dict(tokens=t, implementation=name, uninstrumented=bare, stages=phases,
                   stage_events=events, trace=None)
        if args.mode == "torch":
            marker = Recorder(events=False, nvtx=True, record_functions=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as prof:
                with marker.activate():
                    for _ in range(args.iterations):
                        fn()
                    torch.cuda.synchronize()
            trace = root / f"T{t}-{name}.trace.json"
            prof.export_chrome_trace(str(trace))
            (root / f"T{t}-{name}.operators.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
            row["trace"] = str(trace)
        results.append(row)
        print(f"P0 {name} T={t}: bare median {bare['cuda_median_us']:.2f} us", flush=True)
    (root / "profile.json").write_text(json.dumps(dict(arguments=vars(args),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, shape=dict(hidden=256, ffn=512, experts=8, top_k=2),
        interpretation="Stage CUDA intervals include host-feed idle and instrumentation overhead; parent/child regions overlap. Bare timing is independent.",
        cases=results), indent=2))


if __name__ == "__main__":
    main()
