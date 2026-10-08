"""Invariants, independent scalar references, and numerical derivatives."""
import unittest
import numpy as np
from attentionforge import (scaled_dot_product_attention, stable_softmax, causal_mask,
                            attention_entropy, attention_forward, attention_backward)
from examples.gradient_check import check


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_scalar_reference(self):
        q, k, v = (self.rng.normal(size=s) for s in ((3, 4), (5, 4), (5, 2)))
        out, p = scaled_dot_product_attention(q, k, v)
        reference = np.zeros((3, 2))
        for i in range(3):
            scores = np.array([sum(q[i, d]*k[j, d] for d in range(4))/2 for j in range(5)])
            exps = np.exp(scores - max(scores))
            probs = exps / sum(exps)
            for c in range(2):
                reference[i, c] = sum(probs[j]*v[j, c] for j in range(5))
        np.testing.assert_allclose(out, reference, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(p.sum(axis=-1), 1)

    def test_fully_masked_rows(self):
        q, k, v = [self.rng.normal(size=(3, 4)) for _ in range(3)]
        mask = np.ones((3, 3), bool)
        mask[1] = False
        out, cache = attention_forward(q, k, v, mask=mask)
        np.testing.assert_array_equal(out[1], 0)
        np.testing.assert_array_equal(cache.weights[1], 0)
        dq, dk, dv = attention_backward(np.ones_like(out), cache)
        np.testing.assert_array_equal(dq[1], 0)
        self.assertTrue(all(np.isfinite(x).all() for x in (out, dq, dk, dv)))

    def test_broadcast_key_padding(self):
        q = self.rng.normal(size=(2, 3, 4, 6))
        k = self.rng.normal(size=(2, 3, 5, 6))
        v = self.rng.normal(size=(2, 3, 5, 2))
        mask = np.array([[True, True, False, False, False], [True]*5])[:, None, None, :]
        out, p = scaled_dot_product_attention(q, k, v, mask=mask)
        self.assertEqual(out.shape, (2, 3, 4, 2))
        np.testing.assert_array_equal(p[0, :, :, 2:], 0)

    def test_masked_values_cannot_affect_output(self):
        q = self.rng.normal(size=(2, 4))
        k, v = [self.rng.normal(size=(3, 4)) for _ in range(2)]
        mask = np.array([True, False, True])
        original, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        k[1] = 1000
        v[1] = -1e6
        changed, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        np.testing.assert_allclose(changed, original, atol=1e-13)

    def test_causal_future_invariance(self):
        q, k, v = [self.rng.normal(size=(5, 4)) for _ in range(3)]
        a, p = scaled_dot_product_attention(q, k, v, causal=True)
        k[3:] *= 100
        v[3:] *= -100
        b, _ = scaled_dot_product_attention(q, k, v, causal=True)
        np.testing.assert_array_equal(a[:3], b[:3])
        np.testing.assert_array_equal(p[np.triu_indices(5, 1)], 0)

    def test_causal_offset(self):
        expected = np.array([[True, True, True, False, False], [True, True, True, True, False]])
        np.testing.assert_array_equal(causal_mask(2, 5, query_offset=2), expected)

    def test_temperature_changes_entropy(self):
        x = np.eye(4, dtype=np.float64) * 3
        _, cold = scaled_dot_product_attention(x, x, x, temperature=0.25)
        _, warm = scaled_dot_product_attention(x, x, x, temperature=2)
        self.assertTrue((attention_entropy(warm) > attention_entropy(cold)).all())

    def test_key_permutation_equivariance(self):
        q = self.rng.normal(size=(3, 4))
        k, v = [self.rng.normal(size=(5, 4)) for _ in range(2)]
        perm = np.array([4, 0, 2, 1, 3])
        a, p = scaled_dot_product_attention(q, k, v)
        b, pp = scaled_dot_product_attention(q, k[perm], v[perm])
        np.testing.assert_allclose(a, b, atol=1e-13)
        np.testing.assert_allclose(p[:, perm], pp, atol=1e-13)

    def test_softmax_shift_invariance_and_large_logits(self):
        x = np.array([[1000., 1001., 999.], [-1001., -1000., -999.]])
        np.testing.assert_allclose(stable_softmax(x), stable_softmax(x+5000), atol=1e-13)
        np.testing.assert_allclose(stable_softmax(x).sum(-1), 1)

    def test_float32_preserved(self):
        x = np.eye(3, dtype=np.float32)
        out, p = scaled_dot_product_attention(x, x, x)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(p.dtype, np.float32)

    def test_invalid_inputs(self):
        x = np.eye(3, dtype=np.float64)
        for kwargs in ({"temperature": 0}, {"temperature": np.inf}, {"mask": np.ones((3, 3))},
                       {"query_offset": -1}, {"mask": np.ones((5, 7), bool)}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    scaled_dot_product_attention(x, x, x, **kwargs)
        with self.assertRaises(TypeError):
            scaled_dot_product_attention(x.astype(int), x.astype(int), x.astype(int))
        with self.assertRaises(ValueError):
            scaled_dot_product_attention(x, x[:2], x)
        invalid = x.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            scaled_dot_product_attention(invalid, x, x)

    def test_qkv_numerical_gradients(self):
        report = check()
        self.assertTrue(report["passed"])
        self.assertLess(max(x["max_absolute_error"] for x in report["errors"].values()), 2e-8)

    def test_backward_shape_validation(self):
        x = np.eye(3, dtype=np.float64)
        _, cache = attention_forward(x, x, x)
        with self.assertRaises(ValueError):
            attention_backward(np.ones((2, 3)), cache)


if __name__ == "__main__":
    unittest.main()
