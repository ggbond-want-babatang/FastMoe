"""Inference-only drop-in for Transformers 4.57 OlmoeSparseMoeBlock.

Preserves pretrained router dtype, full softmax and norm_topk_prob; preserves
low-precision SiLU/product rounding and expert-ordered output accumulation.
"""
import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl
from .reference import pack_routes
from .kernels import grouped_mm
from .instrumentation import stage, trace


@triton.jit
def _hf_combine(Y, INV, P, OUT, H: tl.constexpr, TOP_K: tl.constexpr,
                TOTAL: tl.constexpr, B: tl.constexpr):
    ix = tl.program_id(0)*B + tl.arange(0, B)
    token, col = ix // H, ix % H
    acc = tl.zeros((B,), tl.float32)
    # Slots have been sorted by expert ID to match HF's expert loop order.
    for slot in tl.static_range(TOP_K):
        row = tl.load(INV + token*TOP_K + slot, ix < TOTAL, 0)
        p = tl.load(P + token*TOP_K + slot, ix < TOTAL, 0).to(tl.float32)
        v = tl.load(Y + row*H + col, ix < TOTAL, 0).to(tl.float32)
        weighted = (v*p).to(Y.dtype.element_ty).to(tl.float32)
        acc = (acc+weighted).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(OUT + ix, acc, ix < TOTAL)


class TritonOlmoeSparseMoeBlock(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.gate = reference.gate
        self.num_experts = reference.num_experts
        self.top_k = reference.top_k
        self.norm_topk_prob = reference.norm_topk_prob
        if reference.experts[0].config.hidden_act != "silu":
            raise ValueError("Only SwiGLU / SiLU OLMoE supported")
        self.register_buffer("gate_up", torch.stack([
            torch.cat((e.gate_proj.weight.T, e.up_proj.weight.T), dim=1).contiguous()
            for e in reference.experts]), persistent=False)
        self.register_buffer("down", torch.stack([
            e.down_proj.weight.T.contiguous() for e in reference.experts]), persistent=False)
        self.eval()

    @torch.no_grad()
    @trace("hf_triton/full")
    def forward(self, hidden_states):
        if self.training:
            raise RuntimeError("This adapter is inference-only; call eval()")
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        with stage("hf_triton/router"):
            logits = self.gate(x)
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            probs, ids = torch.topk(probs, self.top_k, dim=-1)
            if self.norm_topk_prob:
                probs = probs / probs.sum(-1, keepdim=True)
            probs = probs.to(x.dtype)
            ids, permutation = torch.sort(ids, dim=-1)
            probs = probs.gather(1, permutation).contiguous()
        with stage("hf_triton/dispatch"):
            a, offsets, inverse = pack_routes(x, ids, self.num_experts)
        with stage("hf_triton/gate_up_swiglu"):
            hidden = grouped_mm(a, self.gate_up, offsets, True, round_activation=True)
        with stage("hf_triton/down"):
            values = grouped_mm(hidden, self.down, offsets)
        with stage("hf_triton/combine"):
            out = torch.empty_like(x)
            if out.numel():
                _hf_combine[(triton.cdiv(out.numel(), 256),)](
                    values, inverse, probs, out, x.shape[1], self.top_k, out.numel(), 256)
        return out.reshape(shape), logits
