import unittest
import numpy as np
from attentionforge import MultiHeadAttention, KVCache, apply_rope, sinusoidal_encoding


def numerical_objective_gradient(array, objective, epsilon=1e-6):
    result = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + epsilon
        plus = objective()
        array[index] = original - epsilon
        minus = objective()
        array[index] = original
        result[index] = (plus-minus)/(2*epsilon)
    return result


class MultiHeadTests(unittest.TestCase):
    def test_all_projection_and_cross_input_gradients(self):
        rng = np.random.default_rng(8)
        for rope in (False, True):
            with self.subTest(rope=rope):
                model = MultiHeadAttention(4, 2, seed=8, rope=rope)
                q = rng.normal(size=(1, 2, 4))
                k, v = [rng.normal(size=(1, 3, 4)) for _ in range(2)]
                upstream = rng.normal(size=q.shape)
                out, _, cache = model.forward(q, k, v, return_cache=True)
                dq, dk, dv, grads = model.backward(upstream, cache)
                objective = lambda: float(np.sum(model.forward(q, k, v)[0] * upstream))
                for name, array, grad in [("q", q, dq), ("k", k, dk), ("v", v, dv)] + [
                    (name, array, grads[name]) for name, array in model.params.items()
                ]:
                    numeric = numerical_objective_gradient(array, objective)
                    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8,
                                               err_msg=f"gradient {name}, rope={rope}")

    def test_self_attention_input_gradients_are_summed(self):
        rng = np.random.default_rng(3)
        model = MultiHeadAttention(4, 2, seed=2)
        x = rng.normal(size=(1, 3, 4))
        upstream = rng.normal(size=x.shape)
        _, _, cache = model.forward(x, return_cache=True, causal=True)
        dq, dk, dv, _ = model.backward(upstream, cache)
        objective = lambda: float(np.sum(model.forward(x, causal=True)[0] * upstream))
        numeric = numerical_objective_gradient(x, objective)
        np.testing.assert_allclose(dq+dk+dv, numeric, rtol=1e-5, atol=1e-8)

    def test_full_matches_token_and_chunk_decoding(self):
        for rope in (False, True):
            for dtype in (np.float32, np.float64):
                for chunks in ((1,)*9, (2, 1, 4, 2)):
                    with self.subTest(rope=rope, dtype=dtype, chunks=chunks):
                        x = np.random.default_rng(19).normal(size=(2, 9, 8)).astype(dtype)
                        model = MultiHeadAttention(8, 2, seed=9, dtype=dtype, rope=rope)
                        full = model.forward(x, causal=True)[0]
                        cache = KVCache(2, 2, 9, 4, dtype=dtype)
                        outputs, offset = [], 0
                        for n in chunks:
                            out, weights = model.decode(x[:, offset:offset+n], cache)
                            self.assertEqual(weights.shape[-1], offset+n)
                            outputs.append(out)
                            offset += n
                        atol = 2e-6 if dtype == np.float32 else 1e-12
                        np.testing.assert_allclose(np.concatenate(outputs, axis=1), full,
                                                   rtol=atol, atol=atol)

    def test_cache_overflow_is_atomic(self):
        cache = KVCache(1, 2, 2, 4)
        x = np.ones((1, 2, 1, 4))
        cache.append(x, x)
        before = cache.k[..., :1, :].copy()
        with self.assertRaises(ValueError):
            cache.append(np.ones((1, 2, 2, 4)), np.ones((1, 2, 2, 4)))
        self.assertEqual(cache.length, 1)
        np.testing.assert_array_equal(cache.k[..., :1, :], before)

    def test_cache_reset(self):
        model = MultiHeadAttention(8, 2)
        x = np.ones((1, 2, 8))
        cache = KVCache(1, 2, 2, 4)
        a, _ = model.decode(x, cache)
        cache.reset()
        b, _ = model.decode(x, cache)
        np.testing.assert_array_equal(a, b)

    def test_invalid_decode_does_not_mutate_cache(self):
        model = MultiHeadAttention(8, 2)
        cache = KVCache(1, 2, 3, 4)
        for x, temp in ((np.ones((2, 1, 8)), 1), (np.ones((1, 1, 8)), 0)):
            with self.assertRaises(ValueError):
                model.decode(x, cache, temperature=temp)
            self.assertEqual(cache.length, 0)

    def test_padding_and_causal_compose(self):
        model = MultiHeadAttention(8, 2)
        x = np.ones((1, 4, 8))
        mask = np.array([True, True, False, False])
        _, p, _ = model.forward(x, mask=mask, causal=True)
        np.testing.assert_array_equal(p[..., 2:], 0)
        np.testing.assert_array_equal(p[..., 0, 1:], 0)

    def test_independent_caches_can_coexist(self):
        model = MultiHeadAttention(4, 2)
        x = np.ones((1, 2, 4))
        _, _, first = model.forward(x, return_cache=True)
        a = model.backward(np.ones_like(x), first)[-1]
        model.forward(x*2, return_cache=True)
        b = model.backward(np.ones_like(x), first)[-1]
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])

    def test_constructor_validation(self):
        for args in ((7, 2), (8, 0), (0, 2)):
            with self.assertRaises(ValueError):
                MultiHeadAttention(*args)
        with self.assertRaises(ValueError):
            MultiHeadAttention(6, 2, rope=True)


class PositionTests(unittest.TestCase):
    def test_rotary_norm_and_inverse(self):
        x = np.random.default_rng(12).normal(size=(2, 3, 7, 8))
        y = apply_rope(x, offset=31)
        np.testing.assert_allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1), atol=1e-12)
        np.testing.assert_allclose(apply_rope(y, offset=31, inverse=True), x, atol=1e-12)

    def test_rotary_relative_shift(self):
        q, k = [np.random.default_rng(s).normal(size=(5, 8)) for s in (1, 2)]
        a = apply_rope(q) @ apply_rope(k).T
        b = apply_rope(q, offset=10) @ apply_rope(k, offset=10).T
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_sinusoidal_odd_width_and_offset(self):
        full = sinusoidal_encoding(10, 7)
        part = sinusoidal_encoding(3, 7, offset=5)
        np.testing.assert_array_equal(full[5:8], part)
        np.testing.assert_array_equal(full[0, 0::2], 0)
        np.testing.assert_array_equal(full[0, 1::2], 1)

    def test_rotary_rejects_odd_width(self):
        with self.assertRaises(ValueError):
            apply_rope(np.ones((3, 5)))


if __name__ == "__main__":
    unittest.main()
