# P0 + P3：性能归因与真实 OLMoE 首层接入

这是 v1 MoE 的独立实验副本。服务器目录：

`/home/ggbond/Desktop/MoE/experiments/p0_p3_20261107`

v1 源码、原始测试与性能结果保留在父项目。新副本在共享计算路径加入可关闭阶段标记，并新增真实模型精度语义开关和 Hugging Face OLMoE 适配器。

阅读 [P0_REPORT.md](P0_REPORT.md) 和 [P3_REPORT.md](P3_REPORT.md)。报告中的结论均由原始 JSON、trace 或 Nsight 数据生成。

## 各项优化的独立收益

新增 [MoE 优化消融报告](results/ablation_20261115/REPORT.md)：分别测量 grouped GEMM、投影与激活融合、token 分发、输出合并的独立阶段和完整路径收益。覆盖 20 个输入用例、648 项计时对照，包含 FP16/BF16、真实预训练输入及路由负载分布测试。原始样本、正确性结果和 580 项条件对照均保留在同目录；收益随形状与其他组件而变化。

实验入口为 `ablate_moe.py`，报告生成入口为 `summarize_ablation.py`；复现命令及对照边界见报告。

## 环境

已使用服务器现有的 `/home/ggbond/miniconda3/envs/nanovllm/bin/python`，PyTorch 2.5.1+cu121、Triton 3.1.0、Transformers 4.57.6、safetensors 0.8.0、huggingface_hub 0.36.2。Nsight Systems 2023.2.3。未安装或升级依赖。

## 运行

```bash
cd /home/ggbond/Desktop/MoE/experiments/p0_p3_20261107
bash run_all.sh
```

该入口顺序执行环境检测、权重子集下载或哈希验证、v1 回归、适配器测试、P0 profiling/Nsight、时间线归因、真实层 BF16/FP16 替换与性能测试、报告生成。任何步骤失败立即退出。会覆盖当前最终结果，重要旧结果请先复制保存。

默认 Python 路径可通过 `MOE_PYTHON` 覆盖。权重、Triton/CUDA/Hugging Face 缓存均位于实验目录。下载权重只访问公开固定版本，不执行远程 Python 代码。

P0 独立执行：

```bash
PY=/home/ggbond/miniconda3/envs/nanovllm/bin/python
export TRITON_CACHE_DIR="$PWD/.triton-cache" CUDA_CACHE_PATH="$PWD/.cuda-cache"
"$PY" profile_moe.py --mode torch --iterations 5
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  --output=results/p0/nsys "$PY" profile_moe.py --mode nvtx --iterations 5
"$PY" analyze_traces.py
```

P3 独立执行：

```bash
"$PY" download_layer.py
"$PY" -m tests.test_hf_adapter
"$PY" integrate_olmoe.py --dtype bf16
"$PY" integrate_olmoe.py --dtype fp16 --no-trace
```

`render_figures.py` 是可选的 Matplotlib 静态图绘制脚本，本次在已有 Matplotlib 的桌面环境生成 PNG/SVG，不要求在服务器安装 Matplotlib。`analyze_traces.py` 本身仅依赖 Python 标准库并生成 SVG 时间线。

## 代码结构

- `moe/reference.py`、`moe/kernels.py`：v1 副本，阶段标记默认关闭；grouped GEMM 新增 `round_activation` 开关，默认保留 v1 数值语义。
- `moe/instrumentation.py`：可选 NVTX、record_function 和非逐阶段同步的 CUDA Events。
- `moe/hf_olmoe.py`：推理 adapter，保留真实 Router、不归一化 Top-K、BF16/FP16 激活与累加语义。
- `profile_moe.py`：独立未标记基准、阶段事件、CPU/GPU trace 与 NVTX 捕获。
- `analyze_traces.py`：同时跟踪 Runtime/Driver API 相关 ID，要求所有 GPU kernel 成功归因。
- `integrate_olmoe.py`：加载真实 embedding + decoder layer 0，生成实际 hidden states，比较 MoE 和 decoder 输出及延迟。
- `download_layer.py`：验证 HTTP 206 字节范围，重建所需 202 个张量的独立 safetensors 文件。
- `results/`：日志、退出码、6 份小型 MoE trace、真实层 trace、原生 `.nsys-rep` 和 SQLite、JSON/图表。

## 模型来源与边界

模型为 [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/6d84c48581ece794365f2b8e9cfb043c68ade9c5)，固定 commit `6d84c48581ece794365f2b8e9cfb043c68ade9c5`。权重及使用条件以原始 checkpoint 仓库为准。

P3 包含全部 64 个 expert，真实模型 H=2048/F=1024/Top-K=8。完整 decoder layer 0 保留原始注意力与归一化；只替换 MoE。没有缩小配置、没有随机权重替代，也没有声称完成全模型生成测试。交付包不含预训练权重本体，服务器保留约 997 MiB 的子集及来源清单。
