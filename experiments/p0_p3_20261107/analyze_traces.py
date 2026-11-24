"""Attribute asynchronous GPU kernels to marked CPU launch regions by correlation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import html

COLORS = {"triton/router":"#8c6bb1", "triton/dispatch":"#dd8931",
          "triton/gate_up_swiglu":"#147d73", "triton/down":"#397cbb", "triton/combine":"#ce5276"}


def interval_union(events):
    intervals = sorted((e["ts"], e["ts"]+e.get("dur", 0)) for e in events)
    total, stop = 0., float("-inf")
    for start, end in intervals:
        total += max(0., end-max(start, stop))
        stop = max(stop, end)
    return total


def analyze(path):
    raw = json.loads(path.read_text())
    events = raw["traceEvents"]
    ranges = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
              and (e["name"].startswith(("triton/", "baseline/", "router/", "dispatch/")))]
    full = [e for e in ranges if e["name"] in ("triton/full", "baseline/full")]
    def inside(event, region):
        return (event.get("tid") == region.get("tid") and region["ts"] <= event["ts"] < region["ts"]+region["dur"])
    def containing(event):
        candidates = [r for r in ranges if inside(event, r)]
        return min(candidates, key=lambda r:r["dur"])["name"] if candidates else None
    launches = {}
    runtimes = []
    for event in events:
        # Triton launches through Driver API; ATen mainly uses Runtime API.
        if event.get("cat") in ("cuda_runtime", "cuda_driver") and any(inside(event, r) for r in full):
            runtimes.append(event)
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event
    kernels = []
    gpu_memory = []
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        correlation = event.get("args", {}).get("correlation")
        launch = launches.get(correlation)
        if launch is None:
            continue
        record = dict(name=event["name"], ts=event["ts"], dur=event.get("dur", 0),
                      stage=containing(launch), launch_ts=launch["ts"], correlation=correlation)
        (kernels if event["cat"] == "kernel" else gpu_memory).append(record)
    phase = defaultdict(lambda:dict(kernel_count=0, kernel_sum_us=0.))
    for kernel in kernels:
        phase[kernel["stage"]]["kernel_count"] += 1
        phase[kernel["stage"]]["kernel_sum_us"] += kernel["dur"]
    all_gpu = kernels+gpu_memory
    per_call=[]
    for region in sorted(full,key=lambda r:r["ts"]):
        selected=[e for e in all_gpu if region["ts"] <= e["launch_ts"] < region["ts"]+region["dur"]]
        span_call=max((e["ts"]+e["dur"] for e in selected),default=0)-min((e["ts"] for e in selected),default=0)
        per_call.append(dict(cpu_region_us=region["dur"], device_span_us=span_call,
                             device_active_union_us=interval_union(selected),
                             kernel_sum_us=sum(e["dur"] for e in selected if e in kernels)))
    span = max((e["ts"]+e["dur"] for e in all_gpu), default=0)-min((e["ts"] for e in all_gpu), default=0)
    active = interval_union(all_gpu)
    raw_kernel_count=sum(e.get("cat")=="kernel" for e in events)
    if len(kernels) != raw_kernel_count:
        raise RuntimeError(f"Unattributed GPU kernels in {path}: mapped {len(kernels)} / {raw_kernel_count}")
    return dict(trace=str(path), iterations=len(full), mapped_kernel_count=len(kernels), raw_kernel_count=raw_kernel_count,
        kernel_count_per_call=len(kernels)/max(len(full),1), phases=dict(phase),
        per_call=per_call,
        cuda_runtime_counts=dict(Counter(e["name"] for e in runtimes)),
        memory_operation_names=dict(Counter(e["name"] for e in gpu_memory)),
        device_span_us=span, device_active_union_us=active,
        uncovered_device_span_us=max(0,span-active),
        caveat="Intervals are profiler-perturbed. Uncovered span includes between-iteration host work; GPU kernel sums are not end-to-end latency.",
        kernels=kernels, cpu_ranges=ranges)


def timeline(data, target):
    # Representative invocation by median CPU-region duration, avoiding startup outlier.
    candidates = sorted((r for r in data["cpu_ranges"] if r["name"] == "triton/full"), key=lambda r:r["dur"])
    full = candidates[len(candidates)//2]
    start, end = full["ts"], full["ts"]+full["dur"]
    kernels = [e for e in data["kernels"] if start <= e["launch_ts"] < end]
    ranges = [r for r in data["cpu_ranges"] if r["name"] in COLORS and start <= r["ts"] < end]
    finish = max([end]+[e["ts"]+e["dur"] for e in kernels])
    scale = 880/max(1, finish-start)
    x = lambda t:80+(t-start)*scale
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="420" viewBox="0 0 1080 420">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" fill="#162238">',
             '<text x="30" y="30" font-size="21">T=128 Triton MoE: one profiled invocation</text>',
             '<text x="30" y="54" font-size="12">Actual Kineto timestamps; profiling overhead present. White GPU gaps are not classified as a single cause.</text>',
             '<text x="8" y="112" font-size="12">CPU</text>', '<text x="8" y="222" font-size="12">GPU</text>']
    for i in range(6):
        time = start+(finish-start)*i/5
        xx=x(time)
        parts += [f'<path d="M{xx:.2f} 80 V255" stroke="#e5e9f0"/>',
                  f'<text x="{xx:.2f}" y="280" text-anchor="middle" font-size="12">{time-start:.0f}</text>']
    for r in ranges:
        color=COLORS[r["name"]]
        parts.append(f'<rect x="{x(r["ts"]):.2f}" y="90" width="{max(.7,r["dur"]*scale):.2f}" height="36" fill="{color}"/>')
    for e in kernels:
        name = e["stage"] or ""
        major = "triton/dispatch" if name.startswith("dispatch/") else "triton/router" if name.startswith("router/") else name
        parts.append(f'<rect x="{x(e["ts"]):.2f}" y="200" width="{max(.7,e["dur"]*scale):.2f}" height="36" fill="{COLORS.get(major,"#64748b")}"/>')
    parts.append('<text x="520" y="305" text-anchor="middle" font-size="14">Microseconds since CPU region start</text>')
    for i,(name,color) in enumerate(COLORS.items()):
        xx=30+(i%3)*345
        yy=335+(i//3)*32
        parts += [f'<rect x="{xx}" y="{yy}" width="18" height="14" fill="{color}"/>',
                  f'<text x="{xx+25}" y="{yy+12}" font-size="12">{html.escape(name)}</text>']
    parts += ['</g></svg>']
    target.write_text("\n".join(parts))


def main():
    root=Path("results/p0")
    summaries=[]
    for path in sorted(root.glob("T*-*.trace.json")):
        data=analyze(path)
        if "T128-triton" in path.name:
            timeline(data, root/"timeline-T128.svg")
        summaries.append(data)
    if not summaries or not all(s["mapped_kernel_count"] for s in summaries):
        raise RuntimeError("Missing GPU kernel correlations; inspect trace before reporting")
    (root/"trace-analysis.json").write_text(json.dumps(summaries,indent=2))
    print([(Path(s["trace"]).name,s["mapped_kernel_count"]) for s in summaries])


if __name__ == "__main__":
    main()
