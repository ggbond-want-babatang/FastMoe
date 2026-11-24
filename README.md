# FastMoE Kernels

单 GPU、无 token 丢弃的 MoE **推理**算子实验项目。实现可运行 PyTorch baseline、Triton persistent grouped GEMM 和融合 SwiGLU，提供独立正确性测试与分层 CUDA benchmark。所有结果必须以 `results/` 中实际运行记录为准。

## 语义与张量

- `X[T,H]`；FP32 router `[H,E]`；expert gate/up `[E,H,2F]`；down `[E,F,H]`。
- FP32 router logits，选 Top-K logits 后 softmax，等价于全 expert softmax、Top-K 后重新归一化。无 bias、capacity、dropout、token drop、量化、通信或反向传播。
- 每个 expert：`(SiLU(X @ W_gate) * (X @ W_up)) @ W_down`。
- 投影输出先舍入到 FP16/BF16，再以 FP32 计算 SiLU 和乘法，hidden 回写低精度；down 输出回写低精度；路由加权求和用 FP32，最终输出输入 dtype。
- baseline 支持 CPU FP32；Triton 支持 CUDA FP16/BF16、连续 expert 权重。公开 grouped GEMM 要求 offsets 单调递增、首项 0、末项等于 packed 行数；调用者负责元数据有效性，热路径不读回 GPU 标量。
- 相同 logits 的 Top-K tie-breaking 使用 PyTorch 的行为。

## 实现

`moe/reference.py`：router、expert loop、dispatch 元数据和独立可控路由生成。

`moe/kernels.py`：每个 expert 的不同 M 由 GPU offsets 表达。固定数量 CTA 在 GPU 上遍历 compact tile 空间，空 expert 不分配 tile，支持 M/N/K 尾块 mask。使用 `tl.dot`、FP32 累加，默认 BM=32、BN=64、BK=32，每 SM 最多两个 worker。无需为每个 expert 单独启动 GEMM。

融合版本以两个 dot accumulator 同时计算 Gate/Up，在 epilogue 融合 SiLU 和乘法，避免写回 `[R,2F]` 中间张量。非融合对照使用 grouped gate/up projection + 单独 fused activation。最后的 combine kernel 将 inverse permutation、路由权重乘法、Top-K reduction 融合，每个输出只写一次，避免原子加法。完整路径仍有多个 kernel，**不是整个 MoE 单 kernel 融合**。

Dispatch 使用 PyTorch GPU argsort、scatter 和 gather，未实现专用 routing/dispatch kernel。Grouped GEMM 的调度参考 [Triton 官方 Group GEMM 教程](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)，本项目的 packed offsets、尾块 mask 和 SwiGLU epilogue 为独立实现。

## 在目标服务器运行

已检测可用环境为 `/home/ggbond/miniconda3/envs/nanovllm/bin/python`，无需修改现有环境。确切版本见 `results/environment.json` 和 `results/pip-freeze.txt`。不要在 base 环境运行，也不要根据 `nvidia-smi` 显示的 CUDA 上限替换 PyTorch 环境。

```bash
cd /home/ggbond/Desktop/MoE
export TRITON_CACHE_DIR="$PWD/.triton-cache"
export CUDA_CACHE_PATH="$PWD/.cuda-cache"
PY=/home/ggbond/miniconda3/envs/nanovllm/bin/python
"$PY" environment.py > results/environment.log 2>&1
"$PY" -m tests.test_moe > results/tests.log 2>&1
"$PY" benchmark.py --profile full --tune > results/benchmark.log 2>&1
"$PY" kernel_benchmark.py > results/kernel-benchmark.log 2>&1
"$PY" plot_results.py
"$PY" summarize.py
```

`run.sh` 自动执行上述完整矩阵流程、记录每个步骤退出码并在失败时停止。以 `bash run.sh` 启动。`results/commands.log` 记录实际命令。项目缓存目录位于项目内部，不安装或升级任何依赖。重新运行将覆盖最终结果文件；需要保留旧结果时请先备份 `results/`。

快速 smoke benchmark：

```bash
"$PY" benchmark.py --profile quick --samples 5 --repeats 2 --output results/quick.json
```

完整组合矩阵（6 token 数 × 3 expert 数 × 3 Top-K × 3 人工路由分布，加实际 router E2E）：

```bash
"$PY" benchmark.py --profile full --output results/full.json
```

可调参数 `--tokens 1 8 32 128 512 2048 --hidden 256 --ffn 512 --dtype bf16 --samples 15 --repeats 5`。`--tune` 对 T=512/E=8/K=2/random 的融合 projection 进行 BM={16,32,64}、BN={32,64,128} 九种 tile 的实验。标准 profile 还测试 E=4/K=1 和 E=16/K=4 的 T=128 场景；不把它等同于完整笛卡尔积。

## 测试与 benchmark 解释

测试使用 Python unittest，无 pytest 依赖。涵盖 CPU 独立 dense oracle、CUDA FP32 oracle、FP16/BF16、空 token、空 expert、严重倾斜路由、非整块维度 H=70/F=94、E={4,8,16}、K={1,2,4}、融合消融和 dispatch/inverse 的双向一致性。逐元素误差与相对 L2 双重检查，保存每个比较的误差，不仅打印通过/失败。unittest 的 test method 数不等于参数化比较数。

Benchmark 先逐配置检查正确性并完成 JIT 编译，默认每方法预热 5 次、9 组样本、每组 3 次，记录中位数、P10/P90 和原始样本。

`kernel_benchmark.py` 另外提供更接近纯设备耗时的测量：预先编译并创建 packed 输入，将 32 个相同 grouped kernel launch 捕获到一个 CUDA Graph，计时 graph replay 后除以 32，15 个样本。不含 Python dispatch、分配、routing、排序或 graph capture；反复访问相同权重属于 warm-cache microbenchmark，不能直接与 eager 端到端时延比较。涵盖 T={1,32,512,2048}、E={4,8,16}、K=2、三种路由与两种融合状态，共 72 条测量；此处没有宣称对 PyTorch 的加速比。

- `end_to_end`：输入 X + 原始 router 权重至输出，包含 router、Top-K、dispatch、expert FFN 与 combine。
- `routed_moe`：预先提供 uniform/random/skewed 路由，包含 dispatch、FFN 与 combine，**不包含 router**。skewed 的所有 token 选择同一组 K 个不同 expert。
- `grouped_gemm`：预排列的输入至 gate/up projection，排除 router、dispatch、combine；PyTorch 为 cuBLAS expert loop + concatenate，Triton 为单 grouped kernel；都包含 Python wrapper、分配输出。不是裸 kernel 的理想下界。
- `gate_up_swiglu`：上述 projection 加 SwiGLU，用于评价 epilogue fusion。
- `tile_sweep`：相同工作负载比较九种 tile，并保留 PyTorch 参照。

采用 [CUDA Events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)，同时记录同步后的主机 wall time。eager 模式中 CPU launch 和 baseline 的动态 `torch.where` 同步会使 GPU 出现空闲，CUDA event 区间包含这些空闲；因此不能把该区间称为纯设备算子时间。对照也没有使用 torch.compile 或 CUDA Graph。测试过程只占用此进程的 CUDA allocator 统计；`peak_allocated_bytes` 包含常驻输入与权重，`incremental_peak_bytes` 是相对该方法调用前的增量峰值。与 `nvidia-smi` 的整卡显存用量不同。

吞吐为 tokens/s。有效 TFLOP/s 仅按 FFN GEMM 的理论 FLOP 计数，E2E 不含 router FLOP、激活和 dispatch；不能等同于硬件峰值利用率。

结果：JSON 含原始采样；CSV 供分析；SVG 为独立性能曲线，P10-P90 误差条；`RESULTS.md` 记录实测结论、命令与限制。

## 范围与限制

这是可复现实验实现，未达到生产框架完整性。没有 backward、multi-GPU、expert parallel、模型加载、nano-vLLM 集成、vLLM 对照（目标环境未安装 vLLM）、专用 decode kernel、GPU profiler 或自动最优配置缓存。固定 shape 的 JIT specialization 与显式 tile sweep 不等于完善 autotuning。消费级 GPU 的小工作负载、Python overhead、散热与功耗会影响结果。对 naive expert loop 的收益不能证明优于 vLLM 等成熟实现。不要外推到其他显卡或模型规模。
