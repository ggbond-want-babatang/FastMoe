"""Generate reproducible tables and contrasts from ablate_moe.py raw samples."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def label(c):
    return '/'.join([c['source'], c['dtype'].removeprefix('torch.'),
                     'T'+str(c['tokens']), c['distribution']])


def contrast(methods, baseline, optimized):
    a, b = methods[baseline], methods[optimized]
    x, y = a['cuda_median_us'], b['cuda_median_us']
    rs = [u/v for u, v in zip(a['round_medians_us'], b['round_medians_us'])]
    return dict(baseline=baseline, optimized=optimized, baseline_us=x, optimized_us=y,
                saved_us=x-y, latency_reduction_pct=100*(1-y/x), speedup=x/y,
                round_speedup_min=min(rs), round_speedup_max=max(rs),
                rounds_faster=sum(z>1 for z in rs), rounds=len(rs),
                baseline_p10_us=a['cuda_p10_us'], baseline_p90_us=a['cuda_p90_us'],
                optimized_p10_us=b['cuda_p10_us'], optimized_p90_us=b['cuda_p90_us'])


def full_contrasts(c):
    m = c['fixed_route_paths']['methods']
    rows = []
    def add(factor, context, a, b):
        rows.append(dict(factor=factor, context=context, **contrast(m,a,b)))
    for d, k in itertools.product(('where','sort'), ('torch','triton')):
        add('grouped_gemm',f'dispatch={d},combine={k}',
            f'{d}/loop_tritonact/{k}',f'{d}/grouped_unfused/{k}')
        add('fusion',f'dispatch={d},combine={k}',
            f'{d}/grouped_unfused/{k}',f'{d}/grouped_fused/{k}')
        add('standalone_activation',f'dispatch={d},combine={k}',
            f'{d}/loop_torchact/{k}',f'{d}/loop_tritonact/{k}')
    for g, k in itertools.product(('loop_torchact','loop_tritonact','grouped_unfused','grouped_fused'),
                                  ('torch','triton')):
        add('dispatch',f'compute={g},combine={k}',f'where/{g}/{k}',f'sort/{g}/{k}')
    for d, g in itertools.product(('where','sort'),
                                  ('loop_torchact','loop_tritonact','grouped_unfused','grouped_fused')):
        add('combine',f'dispatch={d},compute={g}',f'{d}/{g}/torch',f'{d}/{g}/triton')
    add('dispatch_combine_joint','compute=grouped_fused',
        'where/grouped_fused/torch','sort/grouped_fused/triton')
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', nargs='?',default='results/ablation_20261115/results.json')
    args = ap.parse_args()
    source = Path(args.input)
    raw = json.loads(source.read_text())
    assert raw['complete'] and len(raw['cases']) == 20
    folder = source.parent
    contrasts, measurements = [], []
    total = 0
    for c in raw['cases']:
        groups = {**c['micro'], 'fixed_route_paths': c['fixed_route_paths']}
        if 'router_included' in c:
            groups['router_included'] = c['router_included']
        for scope, group in groups.items():
            for method,v in group['methods'].items():
                assert v['correctness']['passed']
                assert len(v['cuda_samples_us']) == raw['arguments']['rounds']*raw['arguments']['samples']
                measurements.append(dict(case=label(c),scope=scope,method=method,
                    **{k:v[k] for k in ('cuda_median_us','cuda_p10_us','cuda_p90_us','wall_median_us')},
                    relative_l2=v['correctness']['relative_l2'],max_abs=v['correctness']['max_abs'],
                    round_medians_us=json.dumps(v['round_medians_us'])))
                total += 1
        for row in full_contrasts(c):
            contrasts.append(dict(case=label(c),**row))
    write_csv(folder/'measurements.csv', measurements)
    write_csv(folder/'contrasts.csv', contrasts)
    notes = [
        '# MoE 推理优化：独立阶段基准与完整路径消融',
        '实验执行：2026-11-15；分析整理：2026-11-16（Asia/Shanghai）。本报告不覆盖旧结果。',
        '## 结论与范围',
        '共完成 20 个输入用例、648 项方法计时及 52,488 次正式计时前向（不含预热和正确性检查）。'
        '全部方法通过逐元素与 relative L2 检查。计时覆盖合成小型 MoE 和真实 OLMoE 首层的 MoE 输入，'
        '不是完整 decoder layer 或全模型吞吐测试。',
        '结果支持按输入规模分别描述收益：小型配置 grouped GEMM、融合、分发、合并均有收益；'
        '真实模型中 Gate/Up 的 grouped GEMM 单独收益较小，T=1 时比逐专家矩阵乘法略慢，'
        'Down 投影通常更受益。T=512 时 grouped GEMM 对完整固定路由路径的收益接近零，FP16 略有退化。'
        '真实模型融合在 T=32 的路径收益只有约 1%，T=512 约 13%；不能将总加速全部归因于融合或 grouped GEMM。',
        '## 实验设计与对照',
        '每个用例首先固定同一份路由 ids、probs，测 6 组独立阶段；然后测 '
        '2 种分发 × 4 种专家计算 × 2 种合并 = 16 条完整固定路由路径，另测原生逐专家路径。'
        '这不是 3 个互相独立二元开关的完整析因实验：投影融合嵌套在 grouped GEMM 中。'
        '所有控制路径排除 Router，分发、专家计算、合并及各自输出分配均在计时中。'
        '原生 MoE 与现有优化 MoE 的含 Router 对照单独报告。',
        table(['维度','Baseline','替换后','控制内容'],[
            ['分发','逐专家 torch.where → 拼接 order → packed/offsets/inverse','现有 pack_routes 排序打包','输出布局、专家顺序及元数据相同'],
            ['grouped GEMM','已打包输入上逐专家 PyTorch @ + cat','现有 grouped_mm，激活独立','相同输入、权重、激活 kernel；完整路径同时替换 Gate/Up 与 Down'],
            ['投影激活融合','grouped GEMM → 独立 Triton SwiGLU','现有融合 grouped GEMM','保持相同舍入语义；Down、分发、合并不变'],
            ['合并','按 slot 顺序的 PyTorch gather、乘法、累加','现有 Triton combine / _hf_combine','相同路由权重、slot 顺序及低精度舍入'],
            ['独立激活','PyTorch SwiGLU','独立 Triton SwiGLU','单独控制，避免把激活重写的收益算作投影融合'],
        ]),
        '**Baseline 边界：**逐专家 GEMM 的专家行数在计时外从 GPU 读出并缓存，'
        '不把标量回读同步算进 GEMM。这给 PyTorch 对照提供已知路由结构的优势，'
        '因此完整固定路由循环路径是受控实验，不代表动态服务中的原生实现。'
        '分发对照为了产生同样的 packed/offsets/inverse，会执行额外元数据构造；'
        '它不是原生专家循环中筛选输入的逐指令复刻。合并对照是语义对齐的 PyTorch slot 循环，'
        '不是所有可能 PyTorch 实现中最快者。原生 where+index_add 的整体表现另测，'
        '不能把本报告分发或合并倍数直接称为相对官方实现该阶段的倍数。',
        '## 输入与测量条件',
        '- 合成配置：H=256、F=512、E=8、Top-K=2，FP16/BF16。T=1/128/2048 使用实际随机权重 Router；'
        'T=128 另测 uniform/random/skewed 三种固定路由。随机种子 20261115。skewed 只激活前两个专家。',
        '- 真实配置：固定 checkpoint allenai/OLMoE-1B-7B-0924，revision '
        '6d84c48581ece794365f2b8e9cfb043c68ade9c5；H=2048、F=1024、E=64、Top-K=8，'
        '不归一化 Top-K 概率。B=1，T=1/32/128/512，FP16/BF16。三段手写文本按长度用例轮换、重复并截断；'
        '官方 tokenizer → 真实 embedding → 原始因果 attention/residual/norm → 捕获 MoE 输入。'
        '输入 token IDs、hidden states SHA256、专家计数均记录在 results.json。',
        '- CUDA GPU：'+raw['gpu']+'；PyTorch '+raw['torch']+'；Triton '+raw['triton']+
        '；Transformers '+raw['transformers']+'。保留运行前与最后用例结束后的 nvidia-smi 快照。',
        '- 每个方法先正确性检查和预热 5 次，然后分 3 轮、每轮 9 组、每组 3 次，共 27 个样本、81 次计时前向。'
        '每组打乱各方法的测量顺序，采用 CUDA Event 区间除以重复次数，主表取 27 个样本中位数；'
        'CSV 同时保留 P10/P90、同步 wall time 和每轮中位数。',
        '- 无梯度，无 CUDA Graph，无 profiler；TF32、FP16/BF16 reduced-precision reduction 关闭。'
        '权重打包、原始 attention 输入构造、路由预计算、JIT 编译不计入固定路由路径；'
        '完整含 Router 对照包含每次路由计算。重复执行同一输入及权重，属于 warm-cache 测试。',
        '- 未锁频、未独占 GPU；同一轮的比值范围仅是重复稳定性描述，不是统计置信区间。'
        '3 轮并非 3 次冷启动独立实验。不得用 CUDA Event 时间声称获得纯硬件 kernel 执行时间：'
        '区间可能包含主机发起不及时造成的 GPU 空闲，短算子尤其如此。',
        '## 独立阶段结果',
        '下面的格式为「baseline μs → 优化 μs（倍数）」；倍数 >1 表示优化更快。'
        'Gate/Up 与 Down 分别排除分发、激活和合并；融合阶段比较 grouped_unfused 与 grouped_fused。',
    ]
    stage_pairs=[('gate_up_gemm','torch_loop','grouped'),('down_gemm','torch_loop','grouped'),
                 ('projection_activation','grouped_unfused','grouped_fused'),
                 ('dispatch','expert_where_pack','sort_pack'),('combine','torch','triton')]
    rows=[]
    for c in raw['cases']:
        vals=[]
        for s,a,b in stage_pairs:
            q=contrast(c['micro'][s]['methods'],a,b)
            vals.append(f"{q['baseline_us']:.2f} → {q['optimized_us']:.2f}（{q['speedup']:.2f}×）")
        rows.append([label(c)]+vals)
    notes.append(table(['输入','Gate/Up GEMM','Down GEMM','投影激活融合','分发','合并'],rows))
    notes += ['## 对完整固定路由路径的影响',
              '分别只替换一个环节，其余保持固定。grouped GEMM 与融合对照固定 sort 分发、Triton 合并；'
              '分发和合并对照固定 grouped_fused 专家计算。不同列的 baseline 不同，因此节省时间不能直接相加，倍数不能相乘。'
              '格式为「节省 μs / 路径加速倍数 / 3 轮比值范围」。']
    primary=[('grouped_gemm','sort/loop_tritonact/triton','sort/grouped_unfused/triton'),
             ('fusion','sort/grouped_unfused/triton','sort/grouped_fused/triton'),
             ('dispatch','where/grouped_fused/triton','sort/grouped_fused/triton'),
             ('combine','sort/grouped_fused/torch','sort/grouped_fused/triton')]
    rows=[]
    for c in raw['cases']:
        vals=[]
        for factor,a,b in primary:
            q=contrast(c['fixed_route_paths']['methods'],a,b)
            vals.append(f"{q['saved_us']:.2f} / {q['speedup']:.3f}× / "
                        f"{q['round_speedup_min']:.3f}–{q['round_speedup_max']:.3f}")
        rows.append([label(c)]+vals)
    notes.append(table(['输入','grouped GEMM','投影激活融合','分发','合并'],rows))
    notes.append('![真实 BF16 输入下各改动对完整固定路由路径的影响](ablation-bf16.png)')
    notes += ['## 组合收益与交互',
              'contrasts.csv 为每个用例列出 29 项对照，共 580 项。grouped GEMM、融合、独立激活分别在 4 种上下文中比较；'
              '分发、合并分别在 8 种上下文中比较；另有分发与合并同时替换的对照。'
              '下面列出真实 BF16 输入的上下文敏感性，显示“某项改进的贡献”并非固定常数。']
    rows=[]
    for c in raw['cases']:
        if c['source']!='pretrained_layer0' or c['dtype']!='torch.bfloat16':continue
        cs=full_contrasts(c)
        for factor in ('grouped_gemm','fusion','dispatch','combine'):
            sub=[v for v in cs if v['factor']==factor]
            rows.append([c['tokens'],factor,
                         f"{min(v['speedup'] for v in sub):.3f}–{max(v['speedup'] for v in sub):.3f}×",
                         f"{min(v['saved_us'] for v in sub):.2f}–{max(v['saved_us'] for v in sub):.2f}"])
    notes.append(table(['T','替换项','不同上下文的路径加速范围','节省时间范围 μs'],rows))
    notes += ['## 原生与现有优化实现：含 Router 的完整 MoE',
              '合成对照为 moe_reference 对 moe_triton；真实对照为官方 OlmoeSparseMoeBlock 对 TritonOlmoeSparseMoeBlock。'
              '两者均包含各自完整路由过程，真实模型还包含优化路径为对齐累加顺序执行的 slot 专家排序。'
              '此表是总效果，不用于单独归因。']
    rows=[]
    for c in raw['cases']:
        if 'router_included' not in c:continue
        q=contrast(c['router_included']['methods'],'native','optimized')
        rows.append([label(c),f"{q['baseline_us']:.2f}",f"{q['optimized_us']:.2f}",
                     f"{q['speedup']:.2f}×",f"{q['round_speedup_min']:.2f}–{q['round_speedup_max']:.2f}"])
    notes.append(table(['输入','原生 μs','优化 μs','加速','3 轮范围'],rows))
    notes += ['## 正确性与解读限制',
              '- 共 '+str(total)+' 项方法检查通过。FP16：atol=.004、rtol=.025、relative L2≤.005；'
              'BF16：atol=.035、rtol=.06、relative L2≤.03。分发的 packed、offsets、inverse '
              '与逐专家稳定分组逐元素完全一致。真实模型固定路由参考输出另与官方模块比较。',
              '- 完整路径 grouped GEMM 对照同时改变两次投影，因此需要结合独立 Gate/Up 和 Down 表解释。'
              '独立 grouped GEMM 对照也包含 PyTorch split/cat 及输出分配，不能声称证明 Triton 的矩阵计算内核本身优于 cuBLAS。',
              '- 合并微基准在真实模型上大幅加速，但映射到计算较重的完整路径后收益明显缩小。'
              '分发对照包含逐专家 where 的动态输出同步及同布局元数据构造，不能将其倍数外推给其他 MoE 引擎。',
              '- 真实 T=32 与 T=128 的文本不同，专家负载也不同，因此延迟不保证随 T 单调增加。'
              '这些真实用例仅 B=1、无 KV cache、少量手写文本，不代表完整服务 decode/prefill 或任务质量。',
              '- 各阶段在不同缓存和主机供给状态下单独计时，阶段中位数相加不等于完整路径中位数。'
              '不计算“总收益中 grouped/fusion/dispatch 各占百分之几”这种依赖任意替换顺序的分摊。',
              '## 文件与复现',
              '- `ablate_moe.py`：GPU 消融入口；不改动生产 MoE 内核。',
              '- `summarize_ablation.py`：从原始数据生成本报告与 CSV；没有重新拟合或修改计时数据。',
              '- `results/ablation_20261115/results.json`：原始 27 样本、wall time、测量顺序、正确性、输入与环境快照。',
              '- `measurements.csv`：648 项计时摘要；`contrasts.csv`：580 项完整路径对照。',
              '- `plot_ablation.py`：可选 Matplotlib 绘图；`ablation-bf16.png/.svg` 为真实 BF16 输入的路径收益图。',
              '- `results.json` SHA256：`'+hashlib.sha256(source.read_bytes()).hexdigest()+'`。',
              '```bash\nMOE_PYTHON=/home/ggbond/miniconda3/envs/nanovllm/bin/python\n'
              'TRITON_CACHE_DIR="$PWD/.triton-cache" CUDA_CACHE_PATH="$PWD/.cuda-cache" \\\n'
              '  "$MOE_PYTHON" ablate_moe.py --output results/ablation_new/results.json\n'
              '"$MOE_PYTHON" summarize_ablation.py results/ablation_new/results.json\n```',
              '请为重跑选择新的输出目录，避免覆盖本次原始数据。权重和依赖沿用本项目现有环境。']
    (folder/'REPORT.md').write_text('\n\n'.join(notes)+'\n')
    print('Wrote',folder/'REPORT.md',len(measurements),'measurements',len(contrasts),'contrasts')


if __name__=='__main__':
    main()
