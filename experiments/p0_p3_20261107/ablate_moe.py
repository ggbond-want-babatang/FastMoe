"""Controlled MoE ablations. Writes new results only; requires a CUDA GPU.

Fixed-routing factorial paths use cached CPU expert counts for the torch GEMMs.
This deliberately favors the torch comparator and avoids charging scalar GPU
readbacks to GEMM. Dispatch and combine remain inside each path's timing.
"""
import argparse
import hashlib
import itertools
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors.torch import load_file
from moe.reference import make_weights, route, pack_routes, synthetic_routing, moe_reference
from moe.kernels import grouped_mm, combine, moe_triton
from moe.hf_olmoe import TritonOlmoeSparseMoeBlock, _hf_combine


@triton.jit
def _standalone_swiglu(GU, OUT, N: tl.constexpr, TOTAL: tl.constexpr,
                       ROUND: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // N, i % N
    g = tl.load(GU + row * 2 * N + col, i < TOTAL, 0).to(tl.float32)
    u = tl.load(GU + row * 2 * N + N + col, i < TOTAL, 0).to(tl.float32)
    s = g * tl.sigmoid(g)
    if ROUND:
        s = s.to(GU.dtype.element_ty).to(tl.float32)
    tl.store(OUT + i, s * u, i < TOTAL)


def activation(gu, rounded, backend):
    if backend == "torch":
        g, u = gu.chunk(2, dim=-1)
        if rounded:
            return F.silu(g) * u
        return (F.silu(g.float()) * u.float()).to(gu.dtype)
    out = torch.empty((gu.shape[0], gu.shape[1] // 2), device=gu.device, dtype=gu.dtype)
    _standalone_swiglu[(triton.cdiv(out.numel(), 256),)](
        gu, out, out.shape[1], out.numel(), rounded, 256)
    return out


def loop_mm(a, w, counts):
    # Counts are cached outside timing, not GPU scalar reads in the hot path.
    return torch.cat([part @ w[e] for e, part in
                      enumerate(torch.split(a, counts, dim=0))], dim=0)


def loop_dispatch(x, ids, experts):
    flat = ids.flatten()
    pieces = [torch.where(flat == e)[0] for e in range(experts)]
    order = torch.cat(pieces)
    counts = [p.numel() for p in pieces]
    offsets = torch.tensor([0] + counts, device=x.device, dtype=torch.int64).cumsum(0)
    packed = x[order // ids.shape[1]].contiguous()
    inv = torch.empty_like(order)
    inv.scatter_(0, order, torch.arange(order.numel(), device=x.device))
    return packed, offsets, inv


def merge(y, inv, probs, rounded, backend):
    t, k = probs.shape
    h = y.shape[1]
    if backend == "triton":
        if not rounded:
            return combine(y, inv, probs, (t, h))
        out = torch.empty((t, h), device=y.device, dtype=y.dtype)
        _hf_combine[(triton.cdiv(out.numel(), 256),)](
            y, inv, probs, out, h, k, out.numel(), 256)
        return out
    # Same slot order and rounding as the corresponding Triton kernel.
    if rounded:
        out = torch.zeros((t, h), device=y.device, dtype=y.dtype)
        for s in range(k):
            weighted = (y[inv.reshape(t, k)[:, s]].float() * probs[:, s, None].float()).to(y.dtype)
            out = out + weighted
        return out
    out = torch.zeros((t, h), device=y.device, dtype=torch.float32)
    for s in range(k):
        out = out + y[inv.reshape(t, k)[:, s]].float() * probs[:, s, None]
    return out.to(y.dtype)


def routed_loop(x, ids, probs, gu, down, rounded):
    out = torch.zeros_like(x) if rounded else torch.zeros_like(x, dtype=torch.float32)
    for e in range(gu.shape[0]):
        token, slot = torch.where(ids == e)
        hidden = activation(x[token] @ gu[e], rounded, "torch")
        val = hidden @ down[e]
        weighted = val.float() * probs[token, slot, None].float()
        out.index_add_(0, token, weighted.to(out.dtype))
    return out.to(x.dtype)


def check(actual, expected):
    atol, rtol, limit = ((.004, .025, .005) if actual.dtype == torch.float16
                         else (.035, .06, .03))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    delta = actual.float() - expected.float()
    rel = (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()
    if not torch.isfinite(actual).all().item() or rel > limit:
        raise AssertionError(f"nonfinite or relative L2 {rel}")
    return {"max_abs": delta.abs().max().item(), "relative_l2": rel, "passed": True}


def percentile(xs, q):
    xs = sorted(xs)
    z = q * (len(xs) - 1)
    i = int(z)
    return xs[i] + (xs[min(i+1, len(xs)-1)]-xs[i]) * (z-i)


def measure_group(methods, expected, args, seed):
    errors = {n: check(fn(), expected) for n, fn in methods.items()}
    for fn in methods.values():
        for _ in range(args.warmup):
            fn()
    torch.cuda.synchronize()
    data = {n: {"cuda_samples_us": [], "wall_samples_us": [], "round_medians_us": []}
            for n in methods}
    rng = random.Random(seed)
    order_log = []
    for round_id in range(args.rounds):
        round_data = {n: [] for n in methods}
        for _ in range(args.samples):
            names = list(methods)
            rng.shuffle(names)
            order_log.append(names)
            for n in names:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                tick = time.perf_counter()
                begin.record()
                for _ in range(args.repeats):
                    methods[n]()
                end.record()
                end.synchronize()
                wall = (time.perf_counter()-tick)*1e6/args.repeats
                us = begin.elapsed_time(end)*1000/args.repeats
                data[n]["cuda_samples_us"].append(us)
                data[n]["wall_samples_us"].append(wall)
                round_data[n].append(us)
        for n in methods:
            data[n]["round_medians_us"].append(statistics.median(round_data[n]))
    for n, d in data.items():
        xs = d["cuda_samples_us"]
        d.update(cuda_median_us=statistics.median(xs), cuda_p10_us=percentile(xs, .1),
                 cuda_p90_us=percentile(xs, .9), wall_median_us=statistics.median(d["wall_samples_us"]),
                 correctness=errors[n])
    return {"methods": data, "measurement_orders": order_log}


def real_cases(dtype, tokens):
    from transformers import AutoTokenizer, OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer, OlmoeRotaryEmbedding
    source = Path("weights/olmoe-layer0")
    cfg = OlmoeConfig.from_pretrained(source, local_files_only=True)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        layer = OlmoeDecoderLayer(cfg, layer_idx=0)
    tensors = load_file(str(source / "layer0.safetensors"), device="cuda")
    layer.load_state_dict({k.removeprefix("model.layers.0."): v.to(dtype)
                          for k, v in tensors.items() if k.startswith("model.layers.0.")},
                         strict=True, assign=True)
    embedding = tensors["model.embed_tokens.weight"].to(dtype)
    del tensors
    layer.eval()
    original = layer.mlp
    adapter = TritonOlmoeSparseMoeBlock(original)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    rotary = OlmoeRotaryEmbedding(cfg, device="cuda")
    texts = ["Mixture of Experts routes each token to a small subset of neural network experts. "
             "We compare the original model layer with a grouped matrix multiplication implementation. ",
             "A student has three boxes with eight apples in each box. Explain how to count all the apples. ",
             "CUDA programs execute many threads in parallel. Accurate benchmarking requires warmup and synchronization. "]
    for i, t in enumerate(tokens):
        ids = tokenizer(texts[i % 3] * (t+2), add_special_tokens=False)["input_ids"][:t]
        embedded = F.embedding(torch.tensor([ids], device="cuda"), embedding)
        pos = torch.arange(t, device="cuda")[None]
        mask = torch.full((t,t), torch.finfo(dtype).min, device="cuda", dtype=dtype).triu(1)[None,None]
        captured = {}
        hook = original.register_forward_pre_hook(lambda mod, inputs: captured.update(x=inputs[0].detach()))
        layer(embedded, attention_mask=mask, position_ids=pos, position_embeddings=rotary(embedded, pos),
              use_cache=False, output_router_logits=True)
        hook.remove()
        x = captured["x"].reshape(-1, cfg.hidden_size)
        logits = original.gate(x)
        p, route_ids = F.softmax(logits, -1, dtype=torch.float32).topk(cfg.num_experts_per_tok, -1)
        route_ids, perm = route_ids.sort(-1)
        p = p.to(dtype).gather(1, perm).contiguous()
        yield x, adapter.gate_up, adapter.down, route_ids, p, {
            "source": "pretrained_layer0", "dtype": str(dtype), "tokens": t,
            "hidden": cfg.hidden_size, "ffn": cfg.intermediate_size, "experts": cfg.num_experts,
            "top_k": cfg.num_experts_per_tok, "distribution": "pretrained_router", "input_ids": ids,
            "text_index": i % 3}, original, adapter


def synthetic_cases(dtype, tokens):
    w = make_weights(256, 512, 8, dtype=dtype)
    for t in tokens:
        x = torch.randn(t, 256, device="cuda", dtype=dtype)
        for dist in (["router", "uniform", "random", "skewed"] if t == 128 else ["router"]):
            ids, p = route(x, w.router, 2) if dist == "router" else synthetic_routing(t, 8, 2, dist)
            yield x, w.gate_up, w.down, ids, p, {
                "source": "synthetic", "dtype": str(dtype), "tokens": t, "hidden": 256,
                "ffn": 512, "experts": 8, "top_k": 2, "distribution": dist}, w, None


def run_case(case, args, seed):
    x, gu, down, ids, p, meta, original, adapter = case
    rounded = meta["source"] == "pretrained_layer0"
    e = gu.shape[0]
    a, off, inv = pack_routes(x, ids, e)
    counts = (off[1:]-off[:-1]).tolist()
    for got, want in zip(loop_dispatch(x, ids, e), (a, off, inv)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    expected = routed_loop(x, ids, p, gu, down, rounded)
    if rounded:
        check(expected, original(x[None])[0].reshape_as(x))
    result = dict(**meta, counts=counts, active_experts=sum(c > 0 for c in counts),
                  input_sha256=hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
    micro = {}
    def bench(name, methods, expected):
        micro[name] = measure_group(methods, expected, args, seed+len(micro))
    # Dispatch comparison uses identical stable expert-major layouts.
    bench("dispatch", {"expert_where_pack": lambda: loop_dispatch(x, ids, e)[0],
                       "sort_pack": lambda: pack_routes(x, ids, e)[0]}, a)
    projection = loop_mm(a, gu, counts)
    bench("gate_up_gemm", {"torch_loop": lambda: loop_mm(a, gu, counts),
                           "grouped": lambda: grouped_mm(a, gu, off)}, projection)
    hidden = activation(projection, rounded, "torch")
    bench("activation", {"torch": lambda: activation(projection, rounded, "torch"),
                         "triton": lambda: activation(projection, rounded, "triton")}, hidden)
    bench("projection_activation", {
        "loop_torchact": lambda: activation(loop_mm(a, gu, counts), rounded, "torch"),
        "loop_tritonact": lambda: activation(loop_mm(a, gu, counts), rounded, "triton"),
        "grouped_unfused": lambda: activation(grouped_mm(a, gu, off), rounded, "triton"),
        "grouped_fused": lambda: grouped_mm(a, gu, off, True, round_activation=rounded)}, hidden)
    values = loop_mm(hidden, down, counts)
    bench("down_gemm", {"torch_loop": lambda: loop_mm(hidden, down, counts),
                        "grouped": lambda: grouped_mm(hidden, down, off)}, values)
    bench("combine", {"torch": lambda: merge(values, inv, p, rounded, "torch"),
                      "triton": lambda: merge(values, inv, p, rounded, "triton")},
          merge(values, inv, p, rounded, "torch"))
    result["micro"] = micro
    def path(dispatch_kind, compute, combine_kind):
        aa, offsets, inverse = (pack_routes(x, ids, e) if dispatch_kind == "sort"
                                 else loop_dispatch(x, ids, e))
        if compute == "grouped_fused":
            hh = grouped_mm(aa, gu, offsets, True, round_activation=rounded)
        else:
            proj = (grouped_mm(aa, gu, offsets) if compute == "grouped_unfused"
                    else loop_mm(aa, gu, counts))
            hh = activation(proj, rounded, "torch" if compute == "loop_torchact" else "triton")
        yy = (grouped_mm(hh, down, offsets) if compute.startswith("grouped")
              else loop_mm(hh, down, counts))
        return merge(yy, inverse, p, rounded, combine_kind)
    methods = {"native_routed_loop": lambda: routed_loop(x, ids, p, gu, down, rounded)}
    for d, g, c in itertools.product(("where", "sort"),
            ("loop_torchact", "loop_tritonact", "grouped_unfused", "grouped_fused"),
            ("torch", "triton")):
        methods[f"{d}/{g}/{c}"] = lambda d=d,g=g,c=c: path(d,g,c)
    result["fixed_route_paths"] = measure_group(methods, expected, args, seed+100)
    if rounded:
        native = lambda: original(x[None])[0].reshape_as(x)
        optimized = lambda: adapter(x)[0]
    else:
        native = lambda: moe_reference(x, original, 2)
        optimized = lambda: moe_triton(x, original, 2)
    if rounded or meta["distribution"] == "router":
        result["router_included"] = measure_group({"native": native, "optimized": optimized}, native(), args, seed+200)
    return result


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/ablation_20261115/results.json")
    ap.add_argument("--scope", choices=["all", "synthetic", "real"], default="all")
    ap.add_argument("--dtypes", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"])
    ap.add_argument("--tokens", nargs="+", type=int)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=9)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; no benchmark numbers generated")
    if min(args.warmup, args.rounds, args.samples, args.repeats) < 1 or (args.tokens and min(args.tokens) < 1):
        ap.error("measurement counts and token lengths must be positive")
    import transformers
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def smi():
        return subprocess.run(["nvidia-smi"], text=True, capture_output=True).stdout
    report = {"arguments": vars(args), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "triton": triton.__version__,
              "transformers": transformers.__version__, "platform": platform.platform(),
              "nvidia_smi_before": smi(), "seed": 20261115,
              "semantics": "fixed route paths exclude router; cached host expert counts favor torch loop; native router-included timings separate",
              "cases": []}
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path("moe/kernels.py"), Path("moe/reference.py"), Path("moe/hf_olmoe.py")]}
    for scope in (["synthetic", "real"] if args.scope == "all" else [args.scope]):
        for name in args.dtypes:
            torch.manual_seed(20261115)
            dtype = torch.float16 if name == "fp16" else torch.bfloat16
            cases = (synthetic_cases(dtype, args.tokens or [1,128,2048]) if scope == "synthetic"
                     else real_cases(dtype, args.tokens or [1,32,128,512]))
            for case in cases:
                print("START", {k:v for k,v in case[5].items() if k != "input_ids"}, flush=True)
                row = run_case(case, args, 20261115+len(report["cases"])*1000)
                report["cases"].append(row)
                report["nvidia_smi_latest"] = smi()
                output.write_text(json.dumps(report, indent=2))
                print("DONE", scope, name, row["tokens"], row["distribution"], flush=True)
            del cases
            torch.cuda.empty_cache()
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2))
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
