"""Verify chunked decoding with KV cache and rotary positions."""
import numpy as np
from attentionforge import MultiHeadAttention, KVCache


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 19, 32))
    for rope in (False, True):
        model = MultiHeadAttention(32, 4, seed=9, rope=rope)
        full, _, _ = model.forward(x, causal=True)
        cache = KVCache(2, 4, 19, 8)
        chunks, offset = [], 0
        for length in (1, 4, 2, 5, 7):
            out, _ = model.decode(x[:, offset:offset+length, :], cache)
            chunks.append(out)
            offset += length
        decoded = np.concatenate(chunks, axis=1)
        np.testing.assert_allclose(full, decoded, rtol=1e-12, atol=1e-12)
        print(f"RoPE={rope}: full/chunked max error {np.max(np.abs(full-decoded)):.3e}; "
              f"cache length {cache.length}, preallocated bytes {cache.nbytes:,}")
    print("Every chunk masks its own future tokens while attending to cached keys.")


if __name__ == "__main__":
    main()
