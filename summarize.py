"""Generate an auditable Markdown report from recorded measurements only."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics


def main():
    env = json.loads(Path("results/environment.json").read_text())
    correctness = json.loads(Path("results/correctness.json").read_text())
    bench = json.loads(Path("results/benchmark.json").read_text())
    rows = bench["records"]
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    methods = Counter(r["implementation"] for r in rows)
    comparisons = correctness["comparisons"]
    lines = ["# MoE 远程实测报告", "", f"报告生成时间：{timestamp}（Asia/Shanghai）。", "",
             "## 环境与位置", "",
             "- 服务器项目：`/home/ggbond/Desktop/MoE`；系统桌面入口：`/home/ggbond/桌面/MoE`（符号链接）。",
             f'- GPU：{env["gpu"]["name"]}，显存 {env["gpu"]["total_memory_bytes"]/2**30:.2f} GiB，'
             f'compute capability {env["gpu"]["capability"]}，SM 数 {env["gpu"]["sms"]}。',
             f'- PyTorch {env["torch"]} / CUDA runtime {env["torch_cuda"]} / Triton {env["triton"]}。',
             f'- Python：`{env["executable"]}`。未安装、升级依赖，未修改系统或其他项目文件。',
             "- 缓存限制在项目 `.triton-cache` 和 `.cuda-cache`，没有修改全局缓存配置。", "",
             "## 正确性", "",
             f'- unittest 方法数 {correctness["tests_run"]}，失败 {correctness["failures"]}，'
             f'错误 {correctness["errors"]}，跳过 {correctness["skipped"]}。',
             f'- 记录 {len(comparisons)} 个 CUDA 张量比较；另有 CPU dense oracle、路由及 inverse 检查。']
    for dtype in sorted({c["dtype"] for c in comparisons}):
        cs = [c for c in comparisons if c["dtype"] == dtype]
        lines.append(f'- {dtype}：最大绝对误差 {max(c["max_abs"] for c in cs):.7g}，'
                     f'最大 relative L2 {max(c["relative_l2"] for c in cs):.7g}。')
    lines += ["- 首轮遇到 Triton 3.1 int32/int64 loop-carried variable 编译错误；修复为 int64 边界后重跑全部通过。首次失败日志及最终日志均保留。", "",
              "## 实际执行命令", "", "```bash", "cd /home/ggbond/Desktop/MoE",
              'export TRITON_CACHE_DIR="$PWD/.triton-cache"', 'export CUDA_CACHE_PATH="$PWD/.cuda-cache"',
              "PY=/home/ggbond/miniconda3/envs/nanovllm/bin/python",
              '"$PY" environment.py > results/environment.log 2>&1',
              '"$PY" -m tests.test_moe > results/tests.log 2>&1',
              '"$PY" benchmark.py --profile full --tune > results/benchmark.log 2>&1',
              '"$PY" kernel_benchmark.py > results/kernel-benchmark.log 2>&1',
              '"$PY" plot_results.py', '"$PY" summarize.py', "```", "",
              "完整软件版本见 `results/pip-freeze.txt`，命令和退出码见 `results/commands.log`。", "",
              "## 性能矩阵与计时", "",
              f'- 共 {len(rows)} 条方法/配置测量；方法计数：`{dict(methods)}`。',
              f'- 实际参数：`{json.dumps(bench["arguments"], ensure_ascii=False)}`。',
              "- T={1,8,32,128,512,2048}，E={4,8,16}，K={1,2,4}。每组人工路由覆盖 uniform/random/skewed；E2E 由实际 FP32 router 产生路由。",
              "- 每方法先校验正确性并完成编译，再预热；CUDA event 样本和同步主机 wall time 都保留。使用 eager 执行，没有 CUDA Graph。",
              "- 端到端包括 router、dispatch、FFN 和 combine。grouped GEMM 使用预排列输入，排除路由和 dispatch；其 wrapper 仍包含输出分配及 launch。", "",
              "## 端到端结果（E=8，K=2）", "",
              "| Tokens | PyTorch μs | Triton 非融合 μs | Triton 融合 μs | 融合 P10–P90 μs | 加速比 | 融合 wall μs | 融合显存增量 MiB |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    subset = [r for r in rows if r["scope"] == "end_to_end" and r["experts"] == 8 and r["top_k"] == 2]
    for t in sorted({r["tokens"] for r in subset}):
        group = {r["implementation"]:r for r in subset if r["tokens"] == t}
        a, b, c = (group[n] for n in ("torch_loop", "triton_unfused", "triton_fused"))
        lines.append(f'| {t} | {a["cuda_median_us"]:.2f} | {b["cuda_median_us"]:.2f} | '
                     f'{c["cuda_median_us"]:.2f} | {c["cuda_p10_us"]:.2f}–{c["cuda_p90_us"]:.2f} | '
                     f'{c["speedup_vs_torch"]:.2f}× | {c["wall_median_us"]:.2f} | {c["incremental_peak_bytes"]/2**20:.3f} |')
    lines += ["", "![性能曲线](results/benchmark.svg)", "", "## 分层结果概览", "",
              "下表汇总所有测到的 shape 和路由分布，范围并非同一个工作负载的统计置信区间。", "",
              "| 范围 | Triton 方法 | 配置数 | 最低加速比 | 加速比中位数 | 最高加速比 |", "|---|---|---:|---:|---:|---:|"]
    for scope, method in (("end_to_end", "triton_fused"), ("routed_moe", "triton_fused"),
                          ("grouped_gemm", "triton_grouped"), ("gate_up_swiglu", "triton_fused")):
        vals = [r["speedup_vs_torch"] for r in rows if r["scope"] == scope and r["implementation"] == method]
        lines.append(f'| {scope} | {method} | {len(vals)} | {min(vals):.2f}× | {statistics.median(vals):.2f}× | {max(vals):.2f}× |')
    tune = [r for r in rows if r["scope"] == "tile_sweep" and r["implementation"] != "torch_cublas"]
    if tune:
        best = min(tune, key=lambda r:r["cuda_median_us"])
        lines += ["", "## Tile 实验", "",
                  f'T=512/E=8/K=2/random 的九组 tile 中，本轮最快为 `{best["implementation"]}`，'
                  f'{best["cuda_median_us"]:.2f} μs；默认 BM32/BN64 未根据本轮结果自动更改，避免对测量集过拟合。', "",
                  "| Tile | CUDA 中位数 μs | P10–P90 μs |", "|---|---:|---:|"]
        for r in tune:
            lines.append(f'| {r["implementation"]} | {r["cuda_median_us"]:.2f} | {r["cuda_p10_us"]:.2f}–{r["cuda_p90_us"]:.2f} |')
    kernel_path = Path("results/kernel-benchmark.json")
    if kernel_path.exists():
        kr = json.loads(kernel_path.read_text())["records"]
        lines += ["", "## CUDA Graph 算子耗时", "",
                  f'另外完成 {len(kr)} 条纯设备方向的 microbenchmark，每图 32 个同一 kernel 节点，15 个 replay 样本，按节点均摊。',
                  "预排列输入、GPU offsets、编译与分配均在计时外；反复访问同一权重，是 warm-cache 结果，不可直接替代端到端时延。", "",
                  "| E=8/K=2/random Tokens | Grouped Gate/Up μs | Grouped Gate/Up + SwiGLU μs |", "|---:|---:|---:|"]
        for t in (1, 32, 512, 2048):
            pair = {r["fused_swiglu"]:r for r in kr if r["tokens"] == t and r["experts"] == 8 and r["distribution"] == "random"}
            lines.append(f'| {t} | {pair[False]["median_us"]:.3f} | {pair[True]["median_us"]:.3f} |')
    fused_rows = [r for r in rows if r["scope"] == "end_to_end" and r["implementation"] == "triton_fused"]
    fusion_wins = 0
    for r in fused_rows:
        other = next(q for q in rows if q["scope"] == r["scope"] and q["tokens"] == r["tokens"]
                     and q["experts"] == r["experts"] and q["top_k"] == r["top_k"] and q["implementation"] == "triton_unfused")
        fusion_wins += r["cuda_median_us"] < other["cuda_median_us"]
    lines += ["", "## 解读与限制", "",
              f'- 融合投影在 {len(fused_rows)} 个端到端配置中有 {fusion_wins} 个比非融合版本更快；其余配置没有收益，详见 JSON/CSV。',
              "- PyTorch baseline 使用逐 expert 动态 torch.where，会产生 host/device 同步和多次 launch；加速比包含减少这些开销的收益。CUDA event 区间也包含 CPU 供给不足导致的设备空闲，不等于纯计算 kernel 时长。",
              "- 单 grouped projection 的 PyTorch/cuBLAS 对照使用预先准备的 counts，没有在计时中读取 offsets；仍包含多次 GEMM 和 torch.cat。",
              "- 完整性能矩阵为 FP16；BF16 已做正确性验证，本轮未运行 BF16 性能矩阵。H=256/F=512 是 8GB 显存上的小型实验，不代表真实大型 MoE 模型。",
              "- 固定 worker 数、显式 BM/BN sweep，未实现全自动最优配置搜索、专用 decode kernel、torch.compile、反向、分布式或量化；CUDA Graph 仅用于算子 microbenchmark，端到端未使用。",
              "- 目标环境未安装 vLLM；没有 vLLM 对照、没有 nano-vLLM 集成，不能据此声明优于生产框架。",
              "- 显存为本进程 PyTorch allocator allocated 峰值，不含驱动、Triton module 和其他进程；环境记录保留了测试前的 nvidia-smi。",
              "- 笔记本 GPU 动态功耗、时钟、散热和桌面服务影响波动；没有锁频、独占 GPU 或使用 profiler。没有改动系统设置来控制这些因素。",
              "- 各方法依次测量，未随机化执行顺序；预热次数有限，功耗和缓存状态仍可能影响比较。P10/P90 是本轮样本分位数，不是统计置信区间。",
              "- 没有将选中的路由权重与 GEMM 数值实现融合；router 仍由 PyTorch 完成。Gate/Up+SwiGLU 和 combine 是两处实际融合。", "",
              "## 文件", "",
              "- `results/environment.json`、`results/pip-freeze.txt`：环境。",
              "- `results/tests.log`、`results/correctness.json`：最终测试与逐项误差。",
              "- `results/benchmark.json`、`.csv`：全部测量及原始样本；`.svg`：曲线。",
              "- `results/commands.log`、`results/*.exit`：命令与退出码。",
              "- 首次失败日志独立保存，最终成功状态以最终测试日志和退出码为准。", ""]
    Path("RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"RESULTS.md: {len(rows)} benchmark measurements; {len(comparisons)} correctness comparisons")


if __name__ == "__main__":
    main()
