"""Optional static figure from measured raw data; requires matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_ablation import contrast


def main():
    root = Path('results/ablation_20261115')
    raw = json.loads((root/'results.json').read_text())
    cases = [c for c in raw['cases'] if c['source']=='pretrained_layer0'
             and c['dtype']=='torch.bfloat16']
    factors = [
        ('Grouped GEMM', 'sort/loop_tritonact/triton','sort/grouped_unfused/triton'),
        ('Projection + activation fusion','sort/grouped_unfused/triton','sort/grouped_fused/triton'),
        ('Dispatch','where/grouped_fused/triton','sort/grouped_fused/triton'),
        ('Combine','sort/grouped_fused/torch','sort/grouped_fused/triton')]
    colors = ['#3366aa','#df8439','#2b916f','#9972b0']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(13,5.2))
    for j,(label,a,b) in enumerate(factors):
        ds=[contrast(c['fixed_route_paths']['methods'],a,b) for c in cases]
        pos=[i+(j-1.5)*.19 for i in range(len(cases))]
        axes[0].bar(pos,[d['speedup'] for d in ds],width=.18,color=colors[j],label=label)
        axes[1].bar(pos,[d['saved_us']/1000 for d in ds],width=.18,color=colors[j])
        for i,c in enumerate(cases):
            m=c['fixed_route_paths']['methods']
            rs=[x/y for x,y in zip(m[a]['round_medians_us'],m[b]['round_medians_us'])]
            axes[0].scatter([pos[i]]*len(rs),rs,s=8,color='#222222',zorder=4)
    axes[0].axhline(1,color='#444444',linewidth=1,linestyle='--')
    axes[0].set_ylabel('Baseline latency / changed-path latency')
    axes[0].set_title('One-factor path speedup (dots: 3 round ratios)')
    axes[1].set_ylabel('Baseline latency minus changed-path latency (ms)')
    axes[1].set_title('Absolute latency saved in the full fixed-route path')
    for ax in axes:
        ax.set_xticks(range(len(cases)),[str(c['tokens']) for c in cases])
        ax.set_xlabel('Tokens (batch = 1)')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False,loc='upper right',fontsize=8)
    fig.suptitle('Real OLMoE input, BF16, RTX 4060 Laptop: controlled MoE ablations',fontsize=13)
    fig.text(.5,.015,'Each factor uses its own matched baseline; savings are not additive. Router excluded.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(root/'ablation-bf16.png',dpi=180)
    fig.savefig(root/'ablation-bf16.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
