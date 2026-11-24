"""Optional static report figures; requires matplotlib (rendered on desktop)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "svg.fonttype":"none"})
    root=Path("results")
    data=next(d for d in json.loads((root/"p0/trace-analysis.json").read_text()) if "T128-triton" in d["trace"])
    parents=sorted((r for r in data["cpu_ranges"] if r["name"]=="triton/full"),key=lambda r:r["dur"])
    full=parents[len(parents)//2]
    begin,end=full["ts"],full["ts"]+full["dur"]
    colors={"triton/router":"#8c6bb1","triton/dispatch":"#dd8931",
            "triton/gate_up_swiglu":"#147d73","triton/down":"#397cbb","triton/combine":"#ce5276"}
    fig,ax=plt.subplots(figsize=(12,4.1),layout="constrained")
    for r in data["cpu_ranges"]:
        if r["name"] in colors and begin <= r["ts"] < end:
            ax.broken_barh([(r["ts"]-begin,r["dur"])],(.65,.25),facecolors=colors[r["name"]])
    for r in data["kernels"]:
        if begin <= r["launch_ts"] < end:
            name=r["stage"] or ""
            major="triton/dispatch" if name.startswith("dispatch/") else "triton/router" if name.startswith("router/") else name
            ax.broken_barh([(r["ts"]-begin,r["dur"])],(.15,.25),facecolors=colors.get(major,"#64748b"))
    ax.set_yticks([.775,.275],["CPU launch regions","GPU kernels"])
    ax.set_ylim(0,1.1)
    ax.set_xlabel("Microseconds from CPU region start")
    ax.grid(axis="x",alpha=.15)
    ax.set_axisbelow(True)
    ax.set_title("T=128 MoE timeline | profiler-perturbed, median CPU-duration invocation",loc="left",pad=16)
    ax.legend(handles=[Patch(color=c,label=n.replace("triton/","")) for n,c in colors.items()],
              loc="upper center",bbox_to_anchor=(.5,-.25),ncol=5,frameon=False)
    fig.savefig(root/"p0/timeline-T128.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    d=json.loads((root/"p3/integration-bf16.json").read_text())
    cases=[c for c in d["cases"] if c["batch"]==1]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax,scope,title in zip(axes,["moe","decoder"],["Routed MoE block","Complete decoder layer 0"]):
        for name,label,color in [("hf","Official Transformers","#64748b"),("triton","Triton replacement","#147d73")]:
            rs=[c["timings"][name+"_"+scope] for c in cases]
            values=[r["cuda_median_us"] for r in rs]
            ax.errorbar([c["tokens"] for c in cases],values,
                yerr=[[max(0,v-r["cuda_p10_us"]) for v,r in zip(values,rs)],
                      [max(0,r["cuda_p90_us"]-v) for v,r in zip(values,rs)]],
                color=color,label=label,marker="o",capsize=3)
        ax.set_xscale("log",base=2)
        ax.set_xticks([1,8,32,128,512],[1,8,32,128,512])
        ax.set_xlabel("Sequence tokens (batch=1, no KV cache)")
        ax.set_ylabel("CUDA interval, microseconds")
        ax.set_title(title,loc="left")
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
    fig.suptitle("Pretrained OLMoE | BF16 | H=2048, F=1024, E=64, Top-K=8",fontsize=13)
    fig.savefig(root/"p3/latency-bf16.png",dpi=170,bbox_inches="tight")
    fig.savefig(root/"p3/latency-bf16.svg",bbox_inches="tight")
    plt.close(fig)
    print("Rendered timeline and real-model layer latency figures")


if __name__ == "__main__":
    main()
