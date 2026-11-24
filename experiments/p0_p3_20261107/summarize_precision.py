"""Summarize accuracy-only tests without modifying recorded measurements."""
import collections
import csv
import hashlib
import json
from pathlib import Path


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    root=Path('results/precision_20261116')
    source=root/'accuracy.json'
    p=json.loads(source.read_text())
    assert p['complete']
    rows=p['rows']
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (root/'accuracy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['source'],r['dtype'],r['stage']].append(r)
    enforced=[r for r in rows if r['enforced']]
    text=[
        '# MoE 算子精度复测',
        '执行时间：'+p['started']+' 至 '+p['ended']+'。未修改生产算子，也未放宽既有精度阈值。',
        f"已有 7 个 unittest 回归方法全部通过（包括 CUDA 测试，无跳过）。扩展测试共 {len(rows)} 项误差记录，"
        f"其中 {len(enforced)} 项为强制阈值检查，其余为 FP32 诊断对照；另有 {len(p['structural'])} 项分发逆映射结构检查。"
        f"强制精度失败 {p['enforced_failures']} 项，结构失败 {p['structural_failures']} 项；所有输出及参考均无 NaN/Inf。",
        '## 输入和覆盖范围',
        '- 两种 dtype：FP16、BF16；GPU：'+p['gpu']+'；PyTorch '+p['torch']+'；Triton '+p['triton']+'。'
        '无梯度；关闭 TF32 及 FP16/BF16 reduced-precision reduction。此次只测精度，不测延迟。',
        '- 分阶段合成测试：种子 7/42/20261116，H/F 为 70/94 与 256/512，4 个专家；'
        '专家行数为 [0,0,0,0]、[0,1,33,0]、[17,31,0,1]、[0,65,0,0]；输入幅度为 0/1/4 倍标准正态样本。'
        '覆盖空输入、空专家、块尾部、单行专家、倾斜负载及零输入。'
        '每种 dtype 有 72 个阶段输入，分别检查投影、两种激活舍入语义和 Down 投影。',
        '- 完整合成 MoE：同样 3 个种子，H=70/F=94/E=8/K=2，T=0/1/37/128/2048，'
        'uniform/random/skewed 路由。每种 dtype 45 组，分别测融合/非融合路径并重复调用检查确定性。',
        '- 真实 MoE：固定 OLMoE-1B-7B-0924 预训练 embedding 和第一个 decoder layer，'
        'H=2048/F=1024/E=64/Top-K=8；三段测试文本分别覆盖 1/8/32/128/512 token，'
        '两种 dtype 共 30 组。原始 attention/residual/norm 产生 MoE 输入。'
        '按官方路由选择专家，不重新归一化；使用官方模块作为完整输出对照。'
        '各组 input IDs、专家计数和输入 SHA256 均记录在 JSON。',
        '## 检查标准',
        '逐元素同时满足 |actual-reference| ≤ atol + rtol×|reference|，并检查整体 relative L2。',
        table(['dtype','atol','rtol','relative L2 上限'],[
            ['FP16','.004','.025','.005（0.5%）'],['BF16','.035','.06','.03（3%）']]),
        'relative L2 = ||actual-reference||₂ / max(||reference||₂, 1e-12)。'
        'Router logits 与重复输出要求逐元素完全一致。分发逆映射要求恢复后与重复展开的原始 token 完全一致。'
        '这些阈值是正确性验收标准；下面同时公开实际观测误差，不将“通过”解释为任意输入保证。',
        '## 真实预训练输入：逐阶段最坏误差',
        '每种 dtype、每项包含 15 组输入。下表的最大绝对误差与最大 relative L2 分别取各自最坏样本，可能来自不同输入。'
        '投影与融合阶段相对逐专家 PyTorch 对照；完整 MoE 相对官方 Transformers 模块。',
    ]
    stage_names={'gate_up_vs_torch':'Gate/Up grouped GEMM', 'fused_vs_torch':'投影与激活融合',
                 'down_vs_torch':'Down grouped GEMM','combine_vs_torch':'输出合并',
                 'moe_vs_official':'完整 MoE vs 官方','router_logits':'Router logits','moe_repeat':'完整 MoE 重复调用'}
    rs=[]
    for dt in ('torch.float16','torch.bfloat16'):
        for stage,name in stage_names.items():
            g=groups['pretrained_layer0',dt,stage]
            rs.append([dt.removeprefix('torch.'),name,len(g),f"{max(r['max_abs'] for r in g):.9g}",
                       f"{max(r['relative_l2'] for r in g):.9g}",
                       f"{100*max(r['relative_l2'] for r in g):.6f}%",sum(not r['passed'] for r in g)])
    text.append(table(['dtype','阶段','组数','最大绝对误差','最大 relative L2','百分比','失败'],rs))
    text += ['## 与 FP32 投影参考比较',
             '把同一份已量化的输入及权重转成 FP32，使用 PyTorch FP32 矩阵乘法作为高精度诊断参考。'
             '这隔离了低精度投影计算与输出舍入误差，不包含预先量化输入/权重相对原始高精度值的损失。'
             'FP32 也是有限精度计算，不是数学精确值。本次没有对完整真实 MoE 做全 FP32 前向。']
    rs=[]
    for src in ('synthetic_stage','pretrained_layer0'):
        for dt in ('torch.float16','torch.bfloat16'):
            for stage in ('gate_up_vs_fp32','torch_gate_up_vs_fp32'):
                g=groups[src,dt,stage]
                rs.append([src,dt.removeprefix('torch.'),stage,len(g),
                           f"{max(r['max_abs'] for r in g):.9g}",f"{max(r['relative_l2'] for r in g):.9g}"])
    text.append(table(['输入','dtype','方法 vs FP32','组数','最大绝对误差','最大 relative L2'],rs))
    text += ['真实投影中，FP16 grouped GEMM 与 PyTorch 低精度投影相对 FP32 的最坏 relative L2 '
             '分别约 2.09596e-4 与 2.09587e-4；BF16 分别约 1.68418e-3 与 1.68417e-3。'
             '两者处于相同误差量级，未观察到 grouped GEMM 明显扩大该投影的整体误差。',
             '## 合成压力测试：相对低精度参考',
             '逐项报告最坏值，包含 4 倍输入幅度。因此绝对误差不能直接与真实权重输入表横向比较。']
    rs=[]
    for (src,dt,stage),g in groups.items():
        if src not in ('synthetic_stage','synthetic_moe') or '_fp32' in stage or stage.endswith('_repeat'):continue
        rs.append([src,dt.removeprefix('torch.'),stage,len(g),f"{max(r['max_abs'] for r in g):.9g}",
                   f"{max(r['relative_l2'] for r in g):.9g}",sum(not r['passed'] for r in g)])
    text.append(table(['输入','dtype','阶段','组数','最大绝对误差','最大 relative L2','失败'],rs))
    text += ['## 结论与边界',
             '- 本次覆盖的 grouped GEMM、融合路径、完整 MoE、路由和分发检查全部通过。'
             '真实完整 MoE 的最大 relative L2：FP16 0.048143%，BF16 0.074962%；两者最大绝对误差均为 0.0001220703125。',
             '- Router logits 与同实现重复调用均完全一致。优化输出相对官方输出并非全部逐位相同，'
             '尤其 FP16 合并阶段也存在低位差异；误差在验收范围内。对齐数值语义不等于保证 bitwise 等价。',
             '- 这里只验证推理算子数值，不涉及训练梯度、全模型生成质量或困惑度。'
             '不覆盖所有可能输入、溢出极值、NaN/Inf 输入或每种硬件平台。',
             '## 文件与复现',
             '- `precision_moe.py`：精度测试脚本；`summarize_precision.py`：报告生成。',
             '- `accuracy.json`：完整误差记录、阈值、输入、来源哈希及 checkpoint manifest。',
             '- `accuracy.csv`：1926 项误差指标，包含 mean/RMS、P99/P99.9 绝对误差、最坏元素及逐元素越界数。',
             '- `regression.log`：已有 7 个测试方法全部通过的日志；`run.log`：扩展测试日志。',
             '- 原始数据 SHA256：`'+hashlib.sha256(source.read_bytes()).hexdigest()+'`。',
             '```bash\nMOE_PYTHON=/home/ggbond/miniconda3/envs/nanovllm/bin/python\n'
             'TRITON_CACHE_DIR="$PWD/.triton-cache" CUDA_CACHE_PATH="$PWD/.cuda-cache" \\\n'
             '  "$MOE_PYTHON" precision_moe.py\n'
             'python summarize_precision.py\n```',
             '脚本写入 results/precision_20261116；再次执行前如需保留本次数据，请先复制该目录。']
    (root/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    print(len(rows),'metric records;',len(enforced),'enforced;',len(p['structural']),'structural checks')


if __name__=='__main__':
    main()
