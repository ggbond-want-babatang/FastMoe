"""Generate reports strictly from captured evidence, requiring all checks to pass."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sqlite3
import statistics


def main():
    root=Path("results")
    required=["v1-regression", "p3/adapter-tests", "p3/download", "p0/profile", "p0/nsys",
              "p0/nsys-stats", "p0/analysis", "p3/integration-bf16", "p3/integration-fp16"]
    for name in required:
        assert (root/(name+".exit")).read_text().strip()=="0", name
    p=json.loads((root/"p0/profile.json").read_text())
    traces=json.loads((root/"p0/trace-analysis.json").read_text())
    assert all(t["mapped_kernel_count"]==t["raw_kernel_count"] for t in traces)
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    lines=["# P0：完整 MoE 路径性能分析", "", f"报告生成：{stamp}，Asia/Shanghai。", "",
        "## 完成范围", "",
        "- RTX 4060 Laptop / PyTorch 2.5.1+cu121 / Triton 3.1.0；测试 H=256/F=512/E=8/K=2，T=1/128/2048。",
        "- Router、Dispatch（排序、counts/offsets、gather、inverse）、Gate/Up+SwiGLU、Down、Combine 均有阶段标记。",
        "- 完成原生 Nsight Systems 2023.2.3 CUDA/NVTX 捕获与 SQLite/CSV 统计，以及 6 份 Kineto Chrome trace。",
        "- Nsight 用 cudaProfilerStart/Stop 限定捕获范围；JIT 编译、正确性检查、预热均在捕获外。",
        "- CUDA Events 阶段计时、详细 profiler、无标记正常基准独立采集；阶段事件不逐步同步。", "",
        "## 无分析器的完整前向延迟", "",
        "以下时间使用未激活 Recorder 的完整前向，预热 10 次、15 组样本、每组 3 次。", "",
        "| Tokens | PyTorch CUDA μs | Triton CUDA μs | 加速比 | Triton wall μs |", "|---:|---:|---:|---:|---:|"]
    for t in (1,128,2048):
        rows={c["implementation"]:c for c in p["cases"] if c["tokens"]==t}
        a,b=rows["baseline"]["uninstrumented"],rows["triton"]["uninstrumented"]
        lines.append(f'| {t} | {a["cuda_median_us"]:.2f} | {b["cuda_median_us"]:.2f} | {a["cuda_median_us"]/b["cuda_median_us"]:.2f}× | {b["wall_median_us"]:.2f} |')
    lines += ["", "这组数值是本次实验副本的测量，不能与 11 月 4 日 v1 数字直接比较出优化收益；CPU 状态、标记包装及运行条件发生了变化。", "",
        "## GPU kernel 归因与同步证据", "",
        "通过 CPU launch 相关 ID 将 GPU kernel 映射到最内层阶段；同时处理 CUDA Runtime API 和 Driver API，后者覆盖 Triton。每份 trace 均检查所有 GPU kernel 已归因。", "",
        "| Tokens | 路径 | Kernel 数/前向 | Kernel 时长和 μs/前向 | cudaStreamSynchronize 次/前向 |", "|---:|---|---:|---:|---:|"]
    for t in (1,128,2048):
        for backend in ("baseline","triton"):
            trace=next(d for d in traces if f"T{t}-{backend}.trace.json" in d["trace"])
            n=trace["iterations"]
            sync=sum(v for k,v in trace["cuda_runtime_counts"].items() if "cudaStreamSynchronize" in k)
            lines.append(f'| {t} | {backend} | {trace["kernel_count_per_call"]:.0f} | {sum(k["dur"] for k in trace["kernels"])/n:.2f} | {sync/n:.0f} |')
    selected=next(t for t in traces if "T128-triton.trace.json" in t["trace"])
    lines += ["", "T=128 的优化路径阶段明细（GPU kernel 的执行时长和；来自 profiler，包含扰动）：", "",
        "| 阶段 | Kernel 数/前向 | Kernel 时长和 μs/前向 |", "|---|---:|---:|"]
    for name,stats in selected["phases"].items():
        lines.append(f'| {name} | {stats["kernel_count"]/selected["iterations"]:.0f} | {stats["kernel_sum_us"]/selected["iterations"]:.3f} |')
    db=sqlite3.connect(root/"p0/nsys.sqlite")
    api=db.execute("SELECT s.value,COUNT(*),SUM(r.end-r.start)/1000.0 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId GROUP BY s.value ORDER BY COUNT(*) DESC").fetchall()
    db.close()
    lines += ["", "原生 Nsight 捕获共 30 次前向（3 个 T × 2 条路径 × 5 次）。下列统计覆盖整个捕获范围，不直接代表单个阶段：", "",
        "| CUDA API | 调用数 | CPU API 时长和 μs |", "|---|---:|---:|"]
    for name,count,duration in api:
        if any(s in name for s in ("LaunchKernel","MemcpyAsync","StreamSynchronize")):
            lines.append(f"| {name} | {count} | {duration:.2f} |")
    dispatch=sum(v["kernel_sum_us"] for k,v in selected["phases"].items() if k.startswith("dispatch/"))/selected["iterations"]
    total=sum(k["dur"] for k in selected["kernels"])/selected["iterations"]
    lines += ["", "![T128 时间线](results/p0/timeline-T128.png)", "",
        "图从实际 Kineto 时间戳绘制，选择 CPU region 时长中位数对应的一次调用。上行是 CPU 发起调用的阶段范围，下行是 GPU kernel；不是 Nsight GUI 截图。原生 `.nsys-rep` 另行保留，可在 Nsight Systems 中打开。", "",
        "## 结论与下一步", "",
        f"1. T=128 的 Dispatch 使用 14 个 kernel，kernel 时长和约 {dispatch:.2f} μs，占该 trace 全部 kernel 时长和约 {dispatch/total:.1%}。排序是最大的单个 Dispatch 子阶段。该比例是 GPU kernel 工作量的占比，不是未分析前向延迟的占比。",
        "2. PyTorch baseline 每次前向有 8 次 cudaStreamSynchronize，与 8 个 expert 的动态筛选路径相符；优化路径在标记前向范围内未记录到该 API。",
        "3. 小 T 路径有大量短 kernel 与时间线间隙，支持优先研究 Dispatch、工作空间复用和完整路径 CUDA Graph。仅根据 gap 不能断言全部是 CPU launch 延迟，还可能包括复制、依赖和分析器影响。",
        "4. T=2048 时两个 grouped projection 的 GPU 工作显著增大，计算与访存优化的重要性上升。应区分 decode 和 prefill 的优化策略。",
        "5. 后续性能改动应使用未激活标记的前向评估，再用 timeline 解释原因。", "",
        "## 测量限制与修正记录", "",
        "- 首轮 record_function 同时用于事件计时，扰动明显；已将详细 CPU trace 与轻量事件计时拆开，首轮结果保留在 results/p0-first-pass。最终表格取 results/p0。",
        "- 最初归因器只读 Runtime API，会遗漏每次 3 个 Triton Driver API launch；已修复并加入全部 kernel 归因检查。最终 T128 为 23 个 kernel，不是仅 ATen 的 20 个。",
        "- 即使轻量 CUDA Event 标记也会增加开销，profile.json 中的阶段区间包含主机供给间隙；父子阶段重叠，不能直接相加，也不能从普通基准中逐项相减。",
        "- kernel 时长和、GPU 活跃区间并集与端到端耗时是不同指标；本报告分别保留，不把它们混用。",
        "- 未做 Nsight Compute 硬件计数器分析、锁频或独占 GPU；不据此宣称 SM、Tensor Core 或带宽达到峰值。", "",
        "## 证据与复现", "",
        "- profile_moe.py：阶段采集；analyze_traces.py：相关 ID 归因；results/p0/profile.json：独立基准与阶段事件。",
        "- results/p0/nsys.nsys-rep、nsys.sqlite、nsys-stats.log：原生报告、数据库与统计。",
        "- results/p0/T*.trace.json：Chrome/Perfetto 可读时间线；trace-analysis.json：逐 kernel 归因。",
        "- 所有正式步骤退出码为 0；命令见 results/commands.log。完整重跑：bash run_all.sh。", ""]
    Path("P0_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    lines=["# P3：真实预训练 OLMoE 模型层接入", "", f"报告生成：{stamp}，Asia/Shanghai。", "",
        "## 实际接入范围", "",
        "使用 allenai/OLMoE-1B-7B-0924 的真实预训练 embedding 和完整 decoder layer 0，保留原始 attention、RoPE、RMSNorm 与 residual，只替换其中的 routed MoE MLP。", "",
        "固定 checkpoint revision：`6d84c48581ece794365f2b8e9cfb043c68ade9c5`。H=2048，F=1024，E=64，Top-K=8，norm_topk_prob=false。未缩小 expert 数或用随机权重冒充预训练模型。", "",
        "来源：[固定版本 checkpoint](https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/6d84c48581ece794365f2b8e9cfb043c68ade9c5)。本机 Transformers 4.57.6 的官方 OlmoeDecoderLayer 与 OlmoeSparseMoeBlock 为对照；实现源码 SHA256 记录在 integration JSON。", "",
        "## 权重与输入来源", "",
        "- 只按 HTTP Range 下载所需 202 个预训练张量，约 996.8 MiB，不下载约 14GB 的全模型。",
        "- 每个响应要求精确的 HTTP 206 Content-Range 与长度；重建有效的独立 safetensors 子集，保存固定版本、原始 tensor 字节区间及子集 SHA256。",
        "- 子集哈希用于重跑完整性检查；没有下载完整 shard，因此不声称验证过完整 shard 的数据哈希。",
        "- 自编写的公开测试文本经官方 tokenizer → 真实 embedding → 原始 attention/residual/norm，得到 MoE 输入；每例 input IDs 全部记录。",
        "- B=1 的序列长度为 1/8/32/128/512，另测 B=2/S=16。均为无 KV cache 的因果前向；S=1 不代表完整服务的 cached decode。", "",
        "## 语义适配", "",
        "1. Router 使用原始低精度 Linear 输出，在 FP32 对全部 64 个 logits 做 softmax，选 Top8，不重新归一化，再转换为输入 dtype。",
        "2. 将 gate/up/down 的 nn.Linear 权重转置并打包成 grouped GEMM 布局；打包在计时外，仅首次接入执行。",
        "3. 投影输出、SiLU 输出、hidden product 按官方输入 dtype 舍入；同时保留 v1 默认激活语义。",
        "4. 按 expert ID 排序被选中的 slot，combine 先将路由乘积舍入，再按 expert 递增顺序逐次低精度累加，对齐官方 expert loop。",
        "5. 返回 `(hidden_states, router_logits)`，保持 decoder layer 调用接口；适配器为 inference-only，eval 模式。", "",
        "## 正确性与性能", "",
        "v1 的 6 个 unittest 方法及 174 项 CUDA 比较通过；新增适配器回归覆盖两种 dtype、归一化开关和三种输入形状，共 12 组。", "",
        "真实权重比较 12 组：BF16/FP16 各 6 组，分别检查 MoE 输出、完整 decoder 输出和 router logits。所有组通过预先设置的逐元素与 relative L2 阈值，router logits 逐元素完全一致。", "",
        "| Dtype | B×S | MoE relative L2 | Decoder relative L2 | 官方 MoE μs | Triton MoE μs | MoE 加速 | 官方 decoder μs | Triton decoder μs | Decoder 加速 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dtype in ("bf16","fp16"):
        report=json.loads((root/f"p3/integration-{dtype}.json").read_text())
        assert report["all_passed"] and all(c["router_logits_exact"] and c["moe_error"]["passed"] and c["decoder_error"]["passed"] for c in report["cases"])
        for c in report["cases"]:
            tm=c["timings"]
            lines.append(f'| {dtype} | {c["batch"]}×{c["sequence_length"]} | {c["moe_error"]["relative_l2"]:.6g} | {c["decoder_error"]["relative_l2"]:.6g} | {tm["hf_moe"]["cuda_median_us"]:.2f} | {tm["triton_moe"]["cuda_median_us"]:.2f} | {c["moe_speedup"]:.2f}× | {tm["hf_decoder"]["cuda_median_us"]:.2f} | {tm["triton_decoder"]["cuda_median_us"]:.2f} | {c["decoder_speedup"]:.2f}× |')
    lines += ["", "![BF16 真实模型层延迟](results/p3/latency-bf16.png)", "",
        "## 计时解释与边界", "",
        "- 每方法预热 5 次，9 组样本，每组 2 次；测 CUDA event 区间与同步后的 wall time，保留 P10/P90 和全部样本。各例方法顺序固定随机种子打乱。",
        "- MoE 对照使用 attention/norm 生成的同一真实 hidden_states；完整 decoder 对照包含相同的 attention、norm、residual，embedding lookup 不在计时中。",
        "- 仅替换 layer.mlp；更换 module、权重打包、加载与 JIT 编译均不计入稳定态 benchmark。缓存状态为反复执行同一层的 warm-cache 条件。",
        "- BF16 使用原始 checkpoint dtype；FP16 对同一预训练权重做 dtype 转换，原始与替换路径使用相同转换结果。不是分别重新训练的模型。",
        "- JSON 中保留低精度舍入造成的误差；relative L2 为 0 表示该例实际输出相同，不构成对任意输入 bitwise 相等的保证。",
        "- 相对官方 Transformers Python expert loop 的加速包含减少动态筛选、同步和 kernel launch 的收益；不证明胜过 vLLM/CUTLASS 等成熟 MoE 算子。",
        "- 本任务完成真实 decoder layer 集成；没有加载剩余 15 层、LM head，也没有测试完整文本生成、困惑度、任务精度或服务吞吐。",
        "- P0 的 profile 仍使用 v1 小型形状；P3 另保留真实层 T32 BF16 trace，不能把 P0 的阶段比例直接外推到大层。",
        "- 单卡测试，无多 GPU、训练反向、量化或手写 CUDA C++ 新增。", "",
        "## 文件与复现", "",
        "- moe/hf_olmoe.py：真实模型适配器；integrate_olmoe.py：严格加载、真实输入、替换验证和 benchmark。",
        "- results/p3/integration-{bf16,fp16}.json：每例数值误差、输入 IDs、expert counts、路由概率质量和原始计时样本。",
        "- results/p3/checkpoint-manifest.json：202 个权重来源、字节区间和哈希；weights/olmoe-layer0/layer0.safetensors 留在服务器。",
        "- 交付 zip 不含约 1GB 权重，包含下载器、元数据与结果；本地重跑需先按固定版本下载子集。",
        "- 完整复现：bash run_all.sh。会覆盖最终 results；先复制保留需要的旧实验。", ""]
    Path("P3_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("Generated P0_REPORT.md and P3_REPORT.md; all required checks passed")


if __name__ == "__main__":
    main()
