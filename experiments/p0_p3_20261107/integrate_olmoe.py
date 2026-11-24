"""P3: replace a genuine pretrained OLMoE decoder layer's routed MLP.

Loads embedding + complete decoder layer 0 only. Uses real text, pretrained
embedding, original causal attention and norms to produce actual MLP inputs.
No synthetic/random replacement weights, reduced expert counts, or renormalized
router substitutions are used. This is layer integration, not full generation.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import transformers
from transformers import AutoTokenizer, OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import (OlmoeDecoderLayer,
    OlmoeRotaryEmbedding, OlmoeSparseMoeBlock)
from benchmark import measure
from moe.hf_olmoe import TritonOlmoeSparseMoeBlock
from moe.instrumentation import Recorder, stage


def error_metrics(actual, expected, dtype):
    a, b = actual.float(), expected.float()
    delta = a-b
    relative = (torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(b).clamp_min(1e-12)).item()
    result = dict(max_abs=delta.abs().max().item() if delta.numel() else 0., relative_l2=relative,
                  rms_reference=b.square().mean().sqrt().item() if b.numel() else 0.,
                  finite=bool(torch.isfinite(a).all().item()))
    atol, rtol = (.004, .025) if dtype == torch.float16 else (.035, .06)
    result["tolerances"] = dict(atol=atol, rtol=rtol, relative_l2_limit=.005 if dtype == torch.float16 else .03)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if not result["finite"] or relative > result["tolerances"]["relative_l2_limit"]:
        raise AssertionError(result)
    result["passed"] = True
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/olmoe-layer0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    root = Path("results/p3")
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261107)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    source = Path(args.weights)
    manifest = json.loads((source/"manifest.json").read_text())
    config = OlmoeConfig.from_pretrained(str(source), local_files_only=True)
    config._attn_implementation = "eager"
    assert (config.hidden_size, config.intermediate_size, config.num_experts,
            config.num_experts_per_tok, config.norm_topk_prob) == (2048, 1024, 64, 8, False)
    # Meta construction avoids allocating an extra random 0.8 GiB CPU layer.
    with torch.device("meta"):
        layer = OlmoeDecoderLayer(config, layer_idx=0)
    tensors = load_file(str(source/"layer0.safetensors"), device="cuda")
    prefix = "model.layers.0."
    state = {k[len(prefix):]:v.to(dtype) for k,v in tensors.items() if k.startswith(prefix)}
    incompatible = layer.load_state_dict(state, strict=True, assign=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    embedding = tensors["model.embed_tokens.weight"].to(dtype)
    del state, tensors
    layer.eval()
    original = layer.mlp
    replacement = TritonOlmoeSparseMoeBlock(original)
    rotary = OlmoeRotaryEmbedding(config, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    texts = [
        "Mixture of Experts routes each token to a small subset of neural network experts. "
        "We compare the original model layer with a grouped matrix multiplication implementation. ",
        "A student has three boxes with eight apples in each box. Explain how to count all the apples. ",
        "CUDA programs execute many threads in parallel. Accurate benchmarking requires warmup and synchronization. "]
    config_json = config.to_dict()
    (root/"model-config.json").write_text(json.dumps(config_json, indent=2, default=str))
    report = dict(arguments=vars(args), checkpoint=manifest, transformers=transformers.__version__,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        official_implementation_sha256=hashlib.sha256(inspect.getsource(OlmoeSparseMoeBlock).encode()).hexdigest(),
        layer_class=type(layer).__name__, replacement_class=type(replacement).__name__,
        semantics=dict(norm_topk_prob=False, router="original low-precision Linear, FP32 full softmax, Top8, cast to input dtype",
                       activation="low-precision SiLU and product rounding", accumulation="expert-ID ascending, rounded each add"),
        scope="Pretrained embedding and complete decoder layer 0, original attention and norms; not full model generation",
        cases=[])
    shapes = [(1, t) for t in args.tokens] + [(2, 16)]
    for case_index, (batch, seq) in enumerate(shapes):
        sequences = []
        for b in range(batch):
            text = texts[(case_index+b) % len(texts)]
            ids = tokenizer(text * (seq+2), add_special_tokens=False)["input_ids"][:seq]
            assert len(ids) == seq
            sequences.append(ids)
        input_ids = torch.tensor(sequences, device="cuda")
        embedded = F.embedding(input_ids, embedding)
        positions = torch.arange(seq, device="cuda").expand(batch, -1)
        pos_emb = rotary(embedded, positions)
        mask = torch.full((seq, seq), torch.finfo(dtype).min, device="cuda", dtype=dtype).triu(1)[None,None]
        kwargs = dict(attention_mask=mask, position_ids=positions, position_embeddings=pos_emb,
                      use_cache=False, output_router_logits=True)
        captured = {}
        def capture(module, inputs):
            captured["input"] = inputs[0].detach()
        layer.mlp = original
        handle = original.register_forward_pre_hook(capture)
        expected_decoder = layer(embedded, **kwargs)
        handle.remove()
        moe_input = captured["input"]
        expected_moe, expected_logits = original(moe_input)
        actual_moe, actual_logits = replacement(moe_input)
        moe_error = error_metrics(actual_moe, expected_moe, dtype)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        layer.mlp = replacement
        actual_decoder = layer(embedded, **kwargs)
        decoder_error = error_metrics(actual_decoder[0], expected_decoder[0], dtype)
        torch.testing.assert_close(actual_decoder[-1], expected_decoder[-1], rtol=0, atol=0)
        probabilities = torch.softmax(expected_logits, dim=-1, dtype=torch.float32)
        probs, selected = probabilities.topk(config.num_experts_per_tok, dim=-1)
        counts = torch.bincount(selected.flatten(), minlength=config.num_experts).tolist()
        row = dict(batch=batch, sequence_length=seq, tokens=batch*seq, dtype=args.dtype,
            input_ids=sequences, input_provenance="official tokenizer -> pretrained embedding -> original attention/residual/norm",
            moe_error=moe_error, decoder_error=decoder_error, router_logits_exact=True,
            expert_counts=counts, selected_probability_mass_mean=probs.sum(-1).mean().item(),
            timings={})
        # Randomize method order, adapting the module outside each timing region.
        methods = ["hf_moe", "triton_moe", "hf_decoder", "triton_decoder"]
        random.Random(2026+case_index).shuffle(methods)
        row["measurement_order"] = methods
        for method in methods:
            if method == "hf_moe":
                fn = lambda: original(moe_input)
            elif method == "triton_moe":
                fn = lambda: replacement(moe_input)
            else:
                layer.mlp = original if method == "hf_decoder" else replacement
                fn = lambda: layer(embedded, **kwargs)
            row["timings"][method] = measure(fn, warmup=5, samples=args.samples, repeats=2)
        for scope in ("moe", "decoder"):
            row[scope+"_speedup"] = row["timings"]["hf_"+scope]["cuda_median_us"] / row["timings"]["triton_"+scope]["cuda_median_us"]
        report["cases"].append(row)
        (root/f"integration-{args.dtype}.json").write_text(json.dumps(report, indent=2))
        print(f"P3 {args.dtype} B={batch} S={seq}: MoE L2={moe_error['relative_l2']:.6f} "
              f"MoE {row['moe_speedup']:.2f}x decoder {row['decoder_speedup']:.2f}x", flush=True)
        if not args.no_trace and batch == 1 and seq == 32:
            recorder = Recorder(events=False, nvtx=True, record_functions=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                with recorder.activate():
                    with stage("hf_reference/moe"):
                        original(moe_input)
                    replacement(moe_input)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(root/f"real-layer-T32-{args.dtype}.trace.json"))
    report["all_passed"] = True
    (root/f"integration-{args.dtype}.json").write_text(json.dumps(report, indent=2))
    print("All genuine pretrained layer comparisons passed", flush=True)


if __name__ == "__main__":
    main()
