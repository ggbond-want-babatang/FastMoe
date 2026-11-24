import json
from pathlib import Path
import unittest
import torch
from moe.reference import (make_weights, route, moe_reference, pack_routes,
                           synthetic_routing, swiglu)

METRICS = []


def check(actual, expected, dtype, label):
    atol, rtol = (0.004, 0.02) if dtype == torch.float16 else (0.035, 0.06)
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    rel_l2 = (torch.linalg.vector_norm(diff) /
              torch.linalg.vector_norm(expected.float()).clamp_min(1e-6)).item()
    METRICS.append(dict(case=label, dtype=str(dtype), max_abs=max_abs, relative_l2=rel_l2))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if expected.numel() and rel_l2 > (0.004 if dtype == torch.float16 else 0.035):
        raise AssertionError(f"relative L2 too large: {rel_l2}")


class CPUReferenceTests(unittest.TestCase):
    def test_independent_dense_oracle(self):
        torch.manual_seed(7)
        x = torch.randn(9, 12)
        w = make_weights(12, 20, 4, "cpu", torch.float32)
        ids, probs = route(x, w.router, 2)
        # Independently evaluate every expert for every token, then gather.
        gu = torch.einsum("th,ehf->tef", x, w.gate_up)
        g, u = gu.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(g) * u
        all_y = torch.einsum("tef,efh->teh", hidden, w.down)
        chosen = all_y.gather(1, ids[..., None].expand(-1, -1, 12))
        expected = (chosen * probs[..., None]).sum(1)
        torch.testing.assert_close(moe_reference(x, w, 2), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(probs.sum(-1), torch.ones(9))

    def test_empty_and_invalid(self):
        w = make_weights(8, 12, 4, "cpu", torch.float32)
        self.assertEqual(moe_reference(torch.empty(0, 8), w).shape, (0, 8))
        for k in (0, 5):
            with self.assertRaises(ValueError):
                route(torch.randn(3, 8), w.router, k)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from moe.kernels import grouped_mm, moe_triton, fused_activation
        cls.mm = staticmethod(grouped_mm)
        cls.moe = staticmethod(moe_triton)
        cls.activation = staticmethod(fused_activation)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.manual_seed(42)

    def test_grouped_gemm_and_fused_epilogue(self):
        for dtype in (torch.float16, torch.bfloat16):
            for k, n in ((64, 128), (70, 94)):
                for counts in ([0, 1, 33, 0], [17, 31, 0, 1], [0, 65, 0, 0], [0, 0, 0, 0]):
                    with self.subTest(dtype=dtype, k=k, n=n, counts=counts):
                        a = torch.randn(sum(counts), k, device="cuda", dtype=dtype) / k**0.5
                        weights = torch.randn(4, k, 2*n, device="cuda", dtype=dtype) / k**0.5
                        offsets = torch.tensor([0] + counts, device="cuda", dtype=torch.int64).cumsum(0)
                        expected = torch.empty(sum(counts), 2*n, device="cuda", dtype=dtype)
                        start = 0
                        for e, count in enumerate(counts):
                            expected[start:start+count] = a[start:start+count] @ weights[e]
                            start += count
                        label = f"gemm/{k}/{n}/{counts}"
                        check(self.mm(a, weights, offsets), expected, dtype, label)
                        check(self.mm(a, weights, offsets, True), swiglu(expected), dtype, label+"/fused")
                        check(self.activation(expected), swiglu(expected), dtype, label+"/activation")

    def test_routed_moe(self):
        for dtype in (torch.float16, torch.bfloat16):
            for e in (4, 8, 16):
                w = make_weights(70, 94, e, dtype=dtype)
                for top_k in (1, 2, 4):
                    for dist in ("uniform", "random", "skewed"):
                        for t in (1, 37):
                            label = f"moe/T{t}/E{e}/K{top_k}/{dist}"
                            with self.subTest(label=label, dtype=dtype):
                                x = torch.randn(t, 70, device="cuda", dtype=dtype)
                                routing = synthetic_routing(t, e, top_k, dist)
                                packed, offsets, inv = pack_routes(x, routing[0], e)
                                self.assertEqual(offsets[-1].item(), t*top_k)
                                self.assertTrue(torch.all(offsets[1:] >= offsets[:-1]).item())
                                torch.testing.assert_close(packed[inv], x.repeat_interleave(top_k, dim=0))
                                expected = moe_reference(x, w, top_k, routing)
                                check(self.moe(x, w, top_k, routing), expected, dtype, label)

    def test_end_to_end_and_fusion_ablation(self):
        for dtype in (torch.float16, torch.bfloat16):
            w = make_weights(128, 256, 8, dtype=dtype)
            for t in (0, 8, 128, 512):
                x = torch.randn(t, 128, device="cuda", dtype=dtype)
                expected = moe_reference(x, w, 2)
                for fused in (False, True):
                    with self.subTest(dtype=dtype, t=t, fused=fused):
                        check(self.moe(x, w, 2, fuse_projection=fused), expected, dtype,
                              f"e2e/{t}/fused={fused}")

    def test_cuda_against_fp32_oracle(self):
        from moe.reference import Weights
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(11, 64, device="cuda", dtype=dtype)
            w = make_weights(64, 96, 4, dtype=dtype)
            ids, probs = route(x, w.router, 2)
            wf = Weights(w.router, w.gate_up.float(), w.down.float())
            oracle = moe_reference(x.float(), wf, routing=(ids, probs))
            check(self.moe(x, w, routing=(ids, probs)), oracle.to(dtype), dtype, "fp32-oracle")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path("results").mkdir(exist_ok=True)
    Path("results/correctness.json").write_text(json.dumps({
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "comparisons": METRICS,
    }, indent=2))
    raise SystemExit(not result.wasSuccessful())
