"""Small random regression tests only; pretrained evidence is integrate_olmoe.py."""
import unittest
import torch
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from moe.hf_olmoe import TritonOlmoeSparseMoeBlock
from integrate_olmoe import error_metrics


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AdapterTests(unittest.TestCase):
    def test_official_semantics(self):
        torch.manual_seed(12)
        for dtype in (torch.float16, torch.bfloat16):
            for normalized in (False, True):
                cfg = OlmoeConfig(hidden_size=70, intermediate_size=94, num_experts=4,
                                  num_experts_per_tok=2, norm_topk_prob=normalized)
                ref = OlmoeSparseMoeBlock(cfg).to(device="cuda", dtype=dtype).eval()
                opt = TritonOlmoeSparseMoeBlock(ref)
                for shape in ((1, 1, 70), (2, 17, 70), (1, 0, 70)):
                    with self.subTest(dtype=dtype, normalized=normalized, shape=shape):
                        x = torch.randn(shape, device="cuda", dtype=dtype)
                        with torch.no_grad():
                            y, logits = ref(x)
                            actual, actual_logits = opt(x)
                        error_metrics(actual, y, dtype)
                        torch.testing.assert_close(actual_logits, logits, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
