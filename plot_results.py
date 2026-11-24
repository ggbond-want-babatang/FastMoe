"""Create a self-contained SVG curve, requiring only Python's standard library."""
import html
import json
from pathlib import Path
import math
import sys


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.json")
    data = json.loads(source.read_text())
    rows = [r for r in data["records"] if r["scope"] == "end_to_end" and
            r["experts"] == 8 and r["top_k"] == 2]
    if not rows:
        raise SystemExit("No E=8, K=2 end-to-end measurements")
    width, height = 1000, 620
    left, right, top, bottom = 100, 750, 75, 510
    ts = sorted({r["tokens"] for r in rows})
    ymin = 10**math.floor(math.log10(min(r["cuda_p10_us"] for r in rows)))
    ymax = 10**math.ceil(math.log10(max(r["cuda_p90_us"] for r in rows)))
    def x(t):
        return left + (math.log2(t)-math.log2(ts[0]))/max(1, math.log2(ts[-1])-math.log2(ts[0]))*(right-left)
    def y(v):
        return bottom-(math.log10(v)-math.log10(ymin))/max(1, math.log10(ymax/ymin))*(bottom-top)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<g font-family="Arial, sans-serif" fill="#162238">',
           '<text x="100" y="30" font-size="22">MoE end-to-end latency (E=8, Top-K=2)</text>',
           f'<text x="100" y="53" font-size="13">{html.escape(data["gpu"])} | H={rows[0]["hidden"]}, F={rows[0]["ffn"]} | {rows[0]["dtype"]}</text>']
    for power in range(int(math.log10(ymin)), int(math.log10(ymax))+1):
        val = 10**power
        svg += [f'<path d="M {left} {y(val)} H {right}" stroke="#dfe5ed"/>',
                f'<text x="{left-12}" y="{y(val)+5}" text-anchor="end" font-size="13">{val:g}</text>']
    for t in ts:
        svg += [f'<path d="M {x(t)} {top} V {bottom}" stroke="#eef1f5"/>',
                f'<text x="{x(t)}" y="{bottom+25}" text-anchor="middle" font-size="13">{t}</text>']
    svg += [f'<path d="M {left} {top} V {bottom} H {right}" stroke="#162238" fill="none"/>',
            '<text x="425" y="565" text-anchor="middle" font-size="16">Tokens (log scale)</text>',
            '<text transform="translate(25 300) rotate(-90)" text-anchor="middle" font-size="16">CUDA event latency, us (log scale)</text>']
    for i, (name, color) in enumerate((("torch_loop", "#597084"), ("triton_unfused", "#dd8931"), ("triton_fused", "#147d73"))):
        group = sorted([r for r in rows if r["implementation"] == name], key=lambda r:r["tokens"])
        points = " ".join(f'{x(r["tokens"]):.2f},{y(r["cuda_median_us"]):.2f}' for r in group)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for r in group:
            xx, yy = x(r["tokens"]), y(r["cuda_median_us"])
            svg += [f'<path d="M {xx} {y(r["cuda_p10_us"])} V {y(r["cuda_p90_us"])}" stroke="{color}" stroke-width="2"/>',
                    f'<circle cx="{xx}" cy="{yy}" r="4" fill="{color}"/>']
        yy = 100+i*32
        svg += [f'<path d="M 780 {yy} h 22" stroke="{color}" stroke-width="3"/>',
                f'<text x="810" y="{yy+5}" font-size="13">{name}</text>']
    svg += ['<text x="100" y="603" font-size="12">Median with P10-P90 bars; eager execution, warmup and compilation excluded. Includes routing and dispatch.</text>', '</g></svg>']
    target = source.with_suffix(".svg")
    target.write_text("\n".join(svg))
    print(target)


if __name__ == "__main__":
    main()
