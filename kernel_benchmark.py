"""Device-focused grouped kernels: amortize host launch overhead with CUDA Graph.

Each graph contains 32 independent launches over the same prepacked inputs.
Only graph replay is timed. Sorting, allocations, metadata creation, Python
wrapper work and graph capture are excluded. This is a warm-cache microbenchmark.
"""
import argparse
import json
from pathlib import Path
import statistics
import torch
from moe.reference import make_weights, pack_routes, synthetic_routing, swiglu
from moe.kernels import grouped_mm
from benchmark import packed_reference, assert_correct, quantile


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.nodes < 1 or args.samples < 3:
        parser.error("nodes>=1 and samples>=3 required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    records = []
    for e in (4, 8, 16):
        w = make_weights(256, 512, e)
        for t in (1, 32, 512, 2048):
            x = torch.randn(t, 256, device="cuda", dtype=torch.float16)
            for dist in ("uniform", "random", "skewed"):
                ids, _ = synthetic_routing(t, e, 2, dist)
                a, offsets, _ = pack_routes(x, ids, e)
                counts = (offsets[1:]-offsets[:-1]).tolist()
                expected = packed_reference(a, w.gate_up, counts)
                for fused in (False, True):
                    fn = lambda: grouped_mm(a, w.gate_up, offsets, fused)
                    target = swiglu(expected) if fused else expected
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(5):
                            out = fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    assert_correct(out, target, torch.float16)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(args.nodes):
                            out = fn()
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                    assert_correct(out, target, torch.float16)
                    samples = []
                    for _ in range(args.samples):
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        stop.record()
                        stop.synchronize()
                        samples.append(start.elapsed_time(stop)*1000/args.nodes)
                    row = dict(tokens=t, experts=e, top_k=2, hidden=256, ffn=512,
                               distribution=dist, fused_swiglu=fused, nodes=args.nodes,
                               median_us=statistics.median(samples), p10_us=quantile(samples, .1),
                               p90_us=quantile(samples, .9), samples_us=samples)
                    records.append(row)
                    print(f"kernel T={t} E={e} {dist} fused={fused}: {row['median_us']:.3f} us", flush=True)
                    del graph, out
    Path("results/kernel-benchmark.json").write_text(json.dumps(dict(
        methodology=__doc__, arguments=vars(args), records=records), indent=2))
    print(f"Saved {len(records)} kernel-only measurements")


if __name__ == "__main__":
    main()
