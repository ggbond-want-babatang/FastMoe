"""Accuracy-only stress checks for grouped GEMM, fused SwiGLU and MoE.

Original operator code is not modified. Full sample-level metrics and failures
are retained, including FP32-oracle comparisons that are diagnostic only.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import triton
from ablate_moe import real_cases, loop_mm, activation, merge
from moe.reference import make_weights, pack_routes, synthetic_routing, moe_reference
from moe.kernels import grouped_mm, moe_triton


ROOT = Path('results/precision_20261116')
ROWS = []
STRUCTURAL = []


def record(actual, expected, stage, meta, enforce=True, exact=False):
    a, b = actual.float(), expected.float()
    d = (a-b).abs()
    atol, rtol, limit = ((.004, .025, .005) if actual.dtype == torch.float16
                         else (.035, .06, .03))
    if exact:
        atol, rtol, limit = 0., 0., 0.
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    denom = b.norm().clamp_min(1e-12)
    rel = (d.norm()/denom).item()
    n = d.numel()
    excess = d > atol+rtol*b.abs()
    passed = finite and not excess.any().item() and rel <= limit
    if n:
        q = torch.quantile(d.flatten(), torch.tensor([.99,.999], device=d.device))
        worst = int(d.argmax().item())
        extra = dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                     rms_error=d.square().mean().sqrt().item(), abs_p99=q[0].item(), abs_p999=q[1].item(),
                     exact_fraction=(a==b).float().mean().item(),
                     worst_actual=a.flatten()[worst].item(), worst_reference=b.flatten()[worst].item())
    else:
        extra = dict(max_abs=0.,mean_abs=0.,rms_error=0.,abs_p99=0.,abs_p999=0.,
                     exact_fraction=1.,worst_actual=None,worst_reference=None)
    ROWS.append(dict(**meta,stage=stage,shape=list(actual.shape),elements=n,
                     relative_l2=rel,finite=finite,enforced=enforce,exact_required=exact,
                     passed=passed,violating_elements=int(excess.sum().item()),
                     atol=atol,rtol=rtol,relative_l2_limit=limit,**extra))


def structural(name, flag, meta):
    STRUCTURAL.append(dict(**meta,check=name,passed=bool(flag)))


def synthetic(dtype):
    for seed in (7,42,20261116):
        torch.manual_seed(seed)
        for h,f in ((70,94),(256,512)):
            w=make_weights(h,f,4,dtype=dtype)
            for counts in ([0,0,0,0],[0,1,33,0],[17,31,0,1],[0,65,0,0]):
                off=torch.tensor([0]+counts,device='cuda',dtype=torch.int64).cumsum(0)
                base=torch.randn(sum(counts),h,device='cuda',dtype=dtype)
                for scale in (0.,1.,4.):
                    a=base*scale
                    meta=dict(source='synthetic_stage',dtype=str(dtype),seed=seed,h=h,f=f,
                              counts=counts,input_scale=scale)
                    gu=loop_mm(a,w.gate_up,counts)
                    got_gu=grouped_mm(a,w.gate_up,off)
                    record(got_gu,gu,'gate_up_vs_torch',meta)
                    oracle=loop_mm(a.float(),w.gate_up.float(),counts)
                    record(got_gu,oracle,'gate_up_vs_fp32',meta,enforce=False)
                    record(gu,oracle,'torch_gate_up_vs_fp32',meta,enforce=False)
                    for rounded in (False,True):
                        hidden=activation(gu,rounded,'torch')
                        fused=grouped_mm(a,w.gate_up,off,True,round_activation=rounded)
                        record(fused,hidden,'fused_round'+str(rounded)+'_vs_torch',meta)
                        down=grouped_mm(hidden,w.down,off)
                        ref_down=loop_mm(hidden,w.down,counts)
                        record(down,ref_down,'down_round'+str(rounded)+'_vs_torch',meta)
                        record(down,loop_mm(hidden.float(),w.down.float(),counts),
                               'down_round'+str(rounded)+'_vs_fp32',meta,enforce=False)
        w=make_weights(70,94,8,dtype=dtype)
        for t in (0,1,37,128,2048):
            x=torch.randn(t,70,device='cuda',dtype=dtype)
            for dist in ('uniform','random','skewed'):
                ids,p=synthetic_routing(t,8,2,dist)
                a,off,inv=pack_routes(x,ids,8)
                meta=dict(source='synthetic_moe',dtype=str(dtype),seed=seed,tokens=t,distribution=dist)
                structural('dispatch_inverse',torch.equal(a[inv],x.repeat_interleave(2,0)),meta)
                ref=moe_reference(x,w,2,(ids,p))
                for fused in (False,True):
                    got=moe_triton(x,w,2,(ids,p),fuse_projection=fused)
                    record(got,ref,'moe_fused'+str(fused)+'_vs_torch',meta)
                    repeat=moe_triton(x,w,2,(ids,p),fuse_projection=fused)
                    record(repeat,got,'moe_fused'+str(fused)+'_repeat',meta,exact=True)


def pretrained(dtype):
    # 5 lengths x 3 text rotations: every length sees all 3 texts.
    for index,case in enumerate(real_cases(dtype,[1,8,32,128,512]*3)):
        x,gu,down,ids,p,info,official,adapter=case
        meta={k:v for k,v in info.items() if k!='input_ids'}
        meta['source']='pretrained_layer0'
        meta['input_ids']=info['input_ids']
        meta['hidden_sha256']=hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        a,off,inv=pack_routes(x,ids,gu.shape[0])
        counts=(off[1:]-off[:-1]).tolist()
        meta['counts']=counts
        structural('dispatch_inverse',torch.equal(a[inv],x.repeat_interleave(8,0)),meta)
        expected_gu=loop_mm(a,gu,counts)
        got_gu=grouped_mm(a,gu,off)
        record(got_gu,expected_gu,'gate_up_vs_torch',meta)
        oracle=loop_mm(a.float(),gu.float(),counts)
        record(got_gu,oracle,'gate_up_vs_fp32',meta,enforce=False)
        record(expected_gu,oracle,'torch_gate_up_vs_fp32',meta,enforce=False)
        hidden=activation(expected_gu,True,'torch')
        record(grouped_mm(a,gu,off,True,round_activation=True),hidden,'fused_vs_torch',meta)
        values=loop_mm(hidden,down,counts)
        record(grouped_mm(hidden,down,off),values,'down_vs_torch',meta)
        record(merge(values,inv,p,True,'triton'),merge(values,inv,p,True,'torch'),
               'combine_vs_torch',meta)
        reference,logits=official(x[None])
        got,got_logits=adapter(x[None])
        record(got,reference,'moe_vs_official',meta)
        record(got_logits,logits,'router_logits',meta,exact=True)
        record(adapter(x[None])[0],got,'moe_repeat',meta,exact=True)
        print('REAL',str(dtype),index,'T',x.shape[0],'text',info['text_index'],flush=True)


@torch.no_grad()
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; no CPU substitute results')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    report=dict(started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__,
                rows=ROWS,structural=STRUCTURAL)
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__),Path('moe/kernels.py'),Path('moe/hf_olmoe.py'),Path('ablate_moe.py')]}
    report['checkpoint_manifest']=json.loads(Path('weights/olmoe-layer0/manifest.json').read_text())
    try:
        for dtype in (torch.float16,torch.bfloat16):
            print('SYNTHETIC',dtype,flush=True)
            synthetic(dtype)
            pretrained(dtype)
            (ROOT/'accuracy.json').write_text(json.dumps(report,indent=2))
            torch.cuda.empty_cache()
        report['complete']=True
    finally:
        report['ended']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        report['enforced_failures']=sum(not r['passed'] for r in ROWS if r['enforced'])
        report['structural_failures']=sum(not r['passed'] for r in STRUCTURAL)
        (ROOT/'accuracy.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',len(ROWS),'metrics',report['enforced_failures'],'numerical failures',
          report['structural_failures'],'structural failures',flush=True)
    raise SystemExit(bool(report['enforced_failures'] or report['structural_failures']))


if __name__=='__main__':
    main()
