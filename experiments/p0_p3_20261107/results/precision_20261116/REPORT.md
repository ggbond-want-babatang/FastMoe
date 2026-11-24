# MoE 算子精度复测

执行时间：2026-11-16T13:42:46.443389+08:00 至 2026-11-16T13:42:57.776672+08:00。未修改生产算子，也未放宽既有精度阈值。

已有 7 个 unittest 回归方法全部通过（包括 CUDA 测试，无跳过）。扩展测试共 1926 项误差记录，其中 1290 项为强制阈值检查，其余为 FP32 诊断对照；另有 120 项分发逆映射结构检查。强制精度失败 0 项，结构失败 0 项；所有输出及参考均无 NaN/Inf。

## 输入和覆盖范围

- 两种 dtype：FP16、BF16；GPU：NVIDIA GeForce RTX 4060 Laptop GPU；PyTorch 2.5.1+cu121；Triton 3.1.0。无梯度；关闭 TF32 及 FP16/BF16 reduced-precision reduction。此次只测精度，不测延迟。

- 分阶段合成测试：种子 7/42/20261116，H/F 为 70/94 与 256/512，4 个专家；专家行数为 [0,0,0,0]、[0,1,33,0]、[17,31,0,1]、[0,65,0,0]；输入幅度为 0/1/4 倍标准正态样本。覆盖空输入、空专家、块尾部、单行专家、倾斜负载及零输入。每种 dtype 有 72 个阶段输入，分别检查投影、两种激活舍入语义和 Down 投影。

- 完整合成 MoE：同样 3 个种子，H=70/F=94/E=8/K=2，T=0/1/37/128/2048，uniform/random/skewed 路由。每种 dtype 45 组，分别测融合/非融合路径并重复调用检查确定性。

- 真实 MoE：固定 OLMoE-1B-7B-0924 预训练 embedding 和第一个 decoder layer，H=2048/F=1024/E=64/Top-K=8；三段测试文本分别覆盖 1/8/32/128/512 token，两种 dtype 共 30 组。原始 attention/residual/norm 产生 MoE 输入。按官方路由选择专家，不重新归一化；使用官方模块作为完整输出对照。各组 input IDs、专家计数和输入 SHA256 均记录在 JSON。

## 检查标准

逐元素同时满足 |actual-reference| ≤ atol + rtol×|reference|，并检查整体 relative L2。

| dtype | atol | rtol | relative L2 上限 |
| --- | --- | --- | --- |
| FP16 | .004 | .025 | .005（0.5%） |
| BF16 | .035 | .06 | .03（3%） |

relative L2 = ||actual-reference||₂ / max(||reference||₂, 1e-12)。Router logits 与重复输出要求逐元素完全一致。分发逆映射要求恢复后与重复展开的原始 token 完全一致。这些阈值是正确性验收标准；下面同时公开实际观测误差，不将“通过”解释为任意输入保证。

## 真实预训练输入：逐阶段最坏误差

每种 dtype、每项包含 15 组输入。下表的最大绝对误差与最大 relative L2 分别取各自最坏样本，可能来自不同输入。投影与融合阶段相对逐专家 PyTorch 对照；完整 MoE 相对官方 Transformers 模块。

| dtype | 阶段 | 组数 | 最大绝对误差 | 最大 relative L2 | 百分比 | 失败 |
| --- | --- | --- | --- | --- | --- | --- |
| float16 | Gate/Up grouped GEMM | 15 | 0.0009765625 | 5.01574432e-05 | 0.005016% | 0 |
| float16 | 投影与激活融合 | 15 | 0.00048828125 | 9.93530775e-05 | 0.009935% | 0 |
| float16 | Down grouped GEMM | 15 | 6.10351562e-05 | 3.44767795e-05 | 0.003448% | 0 |
| float16 | 输出合并 | 15 | 0.000122070312 | 0.000454245863 | 0.045425% | 0 |
| float16 | 完整 MoE vs 官方 | 15 | 0.000122070312 | 0.000481431314 | 0.048143% | 0 |
| float16 | Router logits | 15 | 0 | 0 | 0.000000% | 0 |
| float16 | 完整 MoE 重复调用 | 15 | 0 | 0 | 0.000000% | 0 |
| bfloat16 | Gate/Up grouped GEMM | 15 | 0.0078125 | 0.000194040011 | 0.019404% | 0 |
| bfloat16 | 投影与激活融合 | 15 | 0.0009765625 | 0.000137426032 | 0.013743% | 0 |
| bfloat16 | Down grouped GEMM | 15 | 0.000244140625 | 9.23332482e-05 | 0.009233% | 0 |
| bfloat16 | 输出合并 | 15 | 0 | 0 | 0.000000% | 0 |
| bfloat16 | 完整 MoE vs 官方 | 15 | 0.000122070312 | 0.00074961799 | 0.074962% | 0 |
| bfloat16 | Router logits | 15 | 0 | 0 | 0.000000% | 0 |
| bfloat16 | 完整 MoE 重复调用 | 15 | 0 | 0 | 0.000000% | 0 |

## 与 FP32 投影参考比较

把同一份已量化的输入及权重转成 FP32，使用 PyTorch FP32 矩阵乘法作为高精度诊断参考。这隔离了低精度投影计算与输出舍入误差，不包含预先量化输入/权重相对原始高精度值的损失。FP32 也是有限精度计算，不是数学精确值。本次没有对完整真实 MoE 做全 FP32 前向。

| 输入 | dtype | 方法 vs FP32 | 组数 | 最大绝对误差 | 最大 relative L2 |
| --- | --- | --- | --- | --- | --- |
| synthetic_stage | float16 | gate_up_vs_fp32 | 72 | 0.00756454468 | 0.000209858219 |
| synthetic_stage | float16 | torch_gate_up_vs_fp32 | 72 | 0.00756454468 | 0.000209858219 |
| synthetic_stage | bfloat16 | gate_up_vs_fp32 | 72 | 0.0547389984 | 0.00167743734 |
| synthetic_stage | bfloat16 | torch_gate_up_vs_fp32 | 72 | 0.0547389984 | 0.00167743734 |
| pretrained_layer0 | float16 | gate_up_vs_fp32 | 15 | 0.00154733658 | 0.000209595775 |
| pretrained_layer0 | float16 | torch_gate_up_vs_fp32 | 15 | 0.00154733658 | 0.000209587306 |
| pretrained_layer0 | bfloat16 | gate_up_vs_fp32 | 15 | 0.0149531364 | 0.00168417592 |
| pretrained_layer0 | bfloat16 | torch_gate_up_vs_fp32 | 15 | 0.0149531364 | 0.00168417313 |

真实投影中，FP16 grouped GEMM 与 PyTorch 低精度投影相对 FP32 的最坏 relative L2 分别约 2.09596e-4 与 2.09587e-4；BF16 分别约 1.68418e-3 与 1.68417e-3。两者处于相同误差量级，未观察到 grouped GEMM 明显扩大该投影的整体误差。

## 合成压力测试：相对低精度参考

逐项报告最坏值，包含 4 倍输入幅度。因此绝对误差不能直接与真实权重输入表横向比较。

| 输入 | dtype | 阶段 | 组数 | 最大绝对误差 | 最大 relative L2 | 失败 |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic_stage | float16 | gate_up_vs_torch | 72 | 0.00390625 | 8.80617335e-06 | 0 |
| synthetic_stage | float16 | fused_roundFalse_vs_torch | 72 | 0.0078125 | 7.59514933e-06 | 0 |
| synthetic_stage | float16 | down_roundFalse_vs_torch | 72 | 0.001953125 | 1.47771843e-05 | 0 |
| synthetic_stage | float16 | fused_roundTrue_vs_torch | 72 | 0.0078125 | 7.24576603e-06 | 0 |
| synthetic_stage | float16 | down_roundTrue_vs_torch | 72 | 0.015625 | 1.46047823e-05 | 0 |
| synthetic_moe | float16 | moe_fusedFalse_vs_torch | 45 | 0.001953125 | 0.000241661517 | 0 |
| synthetic_moe | float16 | moe_fusedTrue_vs_torch | 45 | 0.001953125 | 0.000241661517 | 0 |
| synthetic_stage | bfloat16 | gate_up_vs_torch | 72 | 0.03125 | 7.0445094e-05 | 0 |
| synthetic_stage | bfloat16 | fused_roundFalse_vs_torch | 72 | 0.0625 | 7.9262958e-05 | 0 |
| synthetic_stage | bfloat16 | down_roundFalse_vs_torch | 72 | 0.03125 | 2.95726095e-05 | 0 |
| synthetic_stage | bfloat16 | fused_roundTrue_vs_torch | 72 | 0.0625 | 0.000237808592 | 0 |
| synthetic_stage | bfloat16 | down_roundTrue_vs_torch | 72 | 0 | 0 | 0 |
| synthetic_moe | bfloat16 | moe_fusedFalse_vs_torch | 45 | 0.0078125 | 0.000137870811 | 0 |
| synthetic_moe | bfloat16 | moe_fusedTrue_vs_torch | 45 | 0.0078125 | 0.000137870811 | 0 |

## 结论与边界

- 本次覆盖的 grouped GEMM、融合路径、完整 MoE、路由和分发检查全部通过。真实完整 MoE 的最大 relative L2：FP16 0.048143%，BF16 0.074962%；两者最大绝对误差均为 0.0001220703125。

- Router logits 与同实现重复调用均完全一致。优化输出相对官方输出并非全部逐位相同，尤其 FP16 合并阶段也存在低位差异；误差在验收范围内。对齐数值语义不等于保证 bitwise 等价。

- 这里只验证推理算子数值，不涉及训练梯度、全模型生成质量或困惑度。不覆盖所有可能输入、溢出极值、NaN/Inf 输入或每种硬件平台。

## 文件与复现

- `precision_moe.py`：精度测试脚本；`summarize_precision.py`：报告生成。

- `accuracy.json`：完整误差记录、阈值、输入、来源哈希及 checkpoint manifest。

- `accuracy.csv`：1926 项误差指标，包含 mean/RMS、P99/P99.9 绝对误差、最坏元素及逐元素越界数。

- `regression.log`：已有 7 个测试方法全部通过的日志；`run.log`：扩展测试日志。

- 原始数据 SHA256：`0a40f85a177a624218d17b15c3c909f00fa59d2a57c75574457b9dcfdf31e431`。

```bash
MOE_PYTHON=/home/ggbond/miniconda3/envs/nanovllm/bin/python
TRITON_CACHE_DIR="$PWD/.triton-cache" CUDA_CACHE_PATH="$PWD/.cuda-cache" \
  "$MOE_PYTHON" precision_moe.py
python summarize_precision.py
```

脚本写入 results/precision_20261116；再次执行前如需保留本次数据，请先复制该目录。
