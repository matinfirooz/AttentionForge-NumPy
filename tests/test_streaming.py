import unittest
import numpy as np
from attentionforge import scaled_dot_product_attention, streaming_attention, attention_storage_elements


class StreamingTests(unittest.TestCase):
    def test_dense_equivalence_with_uneven_blocks(self):
        rng = np.random.default_rng(4)
        for dtype in (np.float32, np.float64):
            q = rng.normal(size=(2, 3, 7, 4)).astype(dtype)
            k = rng.normal(size=(2, 3, 11, 4)).astype(dtype)
            v = rng.normal(size=(2, 3, 11, 5)).astype(dtype)
            for causal in (False, True):
                for qb, kb in ((1, 1), (3, 4), (99, 99)):
                    with self.subTest(dtype=dtype, causal=causal, blocks=(qb, kb)):
                        expected, _ = scaled_dot_product_attention(q, k, v, causal=causal, query_offset=4)
                        actual = streaming_attention(q, k, v, causal=causal, query_offset=4,
                                                     query_block=qb, key_block=kb)
                        tol = 2e-6 if dtype == np.float32 else 1e-12
                        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    def test_fully_masked_tiles_and_rows(self):
        rng = np.random.default_rng(8)
        q = rng.normal(size=(5, 6))
        k, v = [rng.normal(size=(11, 6)) for _ in range(2)]
        mask = rng.random((5, 11)) > 0.6
        mask[:, :4] = False
        mask[2] = False
        expected, _ = scaled_dot_product_attention(q, k, v, mask=mask, temperature=0.6)
        actual = streaming_attention(q, k, v, mask=mask, query_block=3, key_block=4, temperature=0.6)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(actual[2], 0)

    def test_every_row_masked(self):
        x = np.ones((4, 6))
        result = streaming_attention(x, x, x, mask=np.zeros((4, 4), bool), key_block=1)
        np.testing.assert_array_equal(result, 0)

    def test_large_logits_are_stable(self):
        rng = np.random.default_rng(21)
        q, k, v = [rng.normal(size=(9, 4)) for _ in range(3)]
        q, k = q*100, k*100
        expected, _ = scaled_dot_product_attention(q, k, v)
        actual = streaming_attention(q, k, v, query_block=4, key_block=3)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertTrue(np.isfinite(actual).all())

    def test_padding_mask_broadcast(self):
        x = np.random.default_rng(6).normal(size=(2, 3, 5, 4))
        mask = np.array([[True]*5, [True, True, False, False, False]])[:, None, None, :]
        expected, _ = scaled_dot_product_attention(x, x, x, mask=mask, causal=True)
        actual = streaming_attention(x, x, x, mask=mask, causal=True, query_block=2, key_block=3)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_storage_formulas_are_explicit(self):
        state = attention_storage_elements(1024, 1024, 64, query_block=32, key_block=64, batch_heads=2)
        self.assertEqual(state["dense_score_and_weights"], 2*2*1024**2)
        self.assertEqual(state["tiled_score_and_exp"], 2*2*32*64)
        self.assertEqual(state["tiled_running_state"], 2*32*66)

    def test_invalid_block_sizes(self):
        x = np.ones((3, 4))
        for kwargs in ({"query_block": 0}, {"key_block": -1}, {"key_block": 2.5}):
            with self.assertRaises(ValueError):
                streaming_attention(x, x, x, **kwargs)


if __name__ == "__main__":
    unittest.main()
