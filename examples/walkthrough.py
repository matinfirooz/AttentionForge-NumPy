"""Print a tiny, independently understandable attention computation."""
import numpy as np
from attentionforge import scaled_dot_product_attention, causal_mask


def main():
    np.set_printoptions(precision=4, suppress=True)
    q = np.array([[2., 0.], [0., 2.], [1., 1.]])
    k = np.array([[1., 0.], [0., 1.], [1., 1.]])
    v = np.array([[10., 0.], [0., 20.], [30., 30.]])
    scores = q @ k.T / np.sqrt(q.shape[-1])
    output, weights = scaled_dot_product_attention(q, k, v)
    causal_output, causal_weights = scaled_dot_product_attention(q, k, v, causal=True)
    for title, array in (("Q", q), ("K", k), ("V", v),
                         ("1. Scaled similarities Q @ K.T / sqrt(d)", scores),
                         ("2. Softmax probabilities", weights),
                         ("3. Retrieved outputs P @ V", output),
                         ("4. Causal mask: True = allowed", causal_mask(3)),
                         ("5. Causal probabilities", causal_weights),
                         ("6. Causal outputs", causal_output)):
        print(f"\n{title}\n{array}")
    print("\nEach visible row sums to:", weights.sum(axis=-1))


if __name__ == "__main__":
    main()
