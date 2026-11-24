"""Single-device, no-drop Top-K SwiGLU MoE inference reference."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class Weights:
    router: torch.Tensor  # [H, E], FP32
    gate_up: torch.Tensor  # [E, H, 2F], [gate | up]
    down: torch.Tensor  # [E, F, H]


def make_weights(h, f, e, device="cuda", dtype=torch.float16):
    return Weights(torch.randn(h, e, device=device, dtype=torch.float32) / math.sqrt(h),
                   (torch.randn(e, h, 2*f, device=device) / math.sqrt(h)).to(dtype),
                   (torch.randn(e, f, h, device=device) / math.sqrt(f)).to(dtype))


def route(x, router, top_k):
    if not 1 <= top_k <= router.shape[1]:
        raise ValueError("top_k must be in [1, num_experts]")
    # Stable softmax of selected logits equals softmax-all then Top-K renormalization.
    logits = x.float() @ router.float()
    values, indices = torch.topk(logits, top_k, dim=-1)
    return indices, torch.softmax(values, dim=-1)


def validate(x, w):
    if x.ndim != 2 or w.gate_up.ndim != 3 or w.down.ndim != 3:
        raise ValueError("expected X[T,H], gate_up[E,H,2F], down[E,F,H]")
    e, h, two_f = w.gate_up.shape
    if two_f % 2 or x.shape[1] != h or w.down.shape != (e, two_f//2, h) or w.router.shape != (h, e):
        raise ValueError("incompatible MoE shapes")
    if any(t.device != x.device for t in (w.router, w.gate_up, w.down)):
        raise ValueError("all tensors must be on the same device")
    if x.dtype != w.gate_up.dtype or x.dtype != w.down.dtype:
        raise ValueError("X and expert weights must have the same dtype")


def swiglu(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(gate_up.dtype)


@torch.no_grad()
def moe_reference(x, w, top_k=2, routing=None):
    """Deliberately simple expert loop; supports CPU and CUDA, no autograd."""
    validate(x, w)
    ids, probs = route(x, w.router, top_k) if routing is None else routing
    output = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for expert in range(w.gate_up.shape[0]):
        token, slot = torch.where(ids == expert)
        hidden = swiglu(x[token] @ w.gate_up[expert])
        value = hidden @ w.down[expert]
        output.index_add_(0, token, value.float() * probs[token, slot, None])
    return output.to(x.dtype)


def pack_routes(x, ids, experts):
    """GPU-only sorting and metadata construction; no scalar readback or padding."""
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_ids = flat[order]
    # scatter_add avoids bincount's data-dependent output size synchronization.
    counts = torch.zeros(experts, device=x.device, dtype=torch.int64)
    counts.scatter_add_(0, sorted_ids, torch.ones_like(sorted_ids))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    packed = x[order // ids.shape[1]].contiguous()
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=x.device))
    return packed, offsets, inverse


def synthetic_routing(tokens, experts, top_k, distribution, device="cuda"):
    if not 1 <= top_k <= experts:
        raise ValueError("invalid top_k")
    if distribution == "random":
        ids = torch.rand(tokens, experts, device=device).topk(top_k, dim=-1).indices
    elif distribution == "uniform":
        ids = torch.arange(tokens*top_k, device=device).reshape(tokens, top_k) % experts
    elif distribution == "skewed":
        ids = torch.arange(top_k, device=device).expand(tokens, top_k).contiguous()
    else:
        raise ValueError(distribution)
    probs = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1)
    return ids, probs
