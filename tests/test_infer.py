import unittest
import numpy as np
from infer import apply_heads, extract_embedding
from prompts import num, render_camp


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.heads = {"__scale__": 2, "test@1/pick": {
            "kind": "choice", "options": ["a", "b"],
            "weight": [[1, 0], [0, 1]], "bias": [0, 0]}}

    def test_choice(self):
        a = apply_heads(self.heads, "test@1", [1, 0])["pick"]
        self.assertEqual(a["selection"], "a")
        self.assertAlmostEqual(sum(a["probabilities"]), 1)

    def test_normalization(self):
        self.assertEqual(apply_heads(self.heads, "test@1", [1, 0]),
                         apply_heads(self.heads, "test@1", [10, 0]))

    def test_mask(self):
        a = apply_heads(self.heads, "test@1", [1, 0], {"pick": [False, True]})
        self.assertEqual(a["pick"]["selection"], "b")

    def test_invalid_masks(self):
        for m in ([False, False], [True], ["false", "true"]):
            with self.assertRaises(ValueError):
                apply_heads(self.heads, "test@1", [1, 0], {"pick": m})

    def test_multi(self):
        self.heads["test@1/pick"]["kind"] = "multi"
        out = apply_heads(self.heads, "test@1", [1, -1])["pick"]
        self.assertEqual(out["selection"], ["a"])

    def test_embedding_shapes(self):
        for v in ([{"embedding": [[1, 2]]}], {"data": [{"embedding": [1, 2]}]}):
            np.testing.assert_equal(extract_embedding(v), [1, 2])

    def test_bad_hidden(self):
        for h in ([0, 0], [float("nan"), 0], [1, 2, 3]):
            with self.assertRaises(ValueError):
                apply_heads(self.heads, "test@1", h)

    def test_unknown_channel(self):
        with self.assertRaises(ValueError):
            apply_heads(self.heads, "missing", [1, 0])

    def test_prompt(self):
        self.assertEqual(render_camp({"battle": {}}),
            "<decision> camp_assessment@1\nscore=50 tempo=0.5 survival=0.5 tactics=0.5\n<decide>")
        self.assertEqual(num(-0.125), "-0.13")


if __name__ == "__main__":
    unittest.main()
