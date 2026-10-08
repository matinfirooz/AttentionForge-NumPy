"""Finite-difference attention gradients. Independent of the backward formula."""
import json
from pathlib import Path
import numpy as np
from attentionforge import attention_forward, attention_backward


def check(*, seed=7, epsilon=1e-6):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(2, 3, 4))
    k = rng.normal(size=(2, 5, 4))
    v = rng.normal(size=(2, 5, 3))
    upstream = rng.normal(size=(2, 3, 3))
    mask = rng.random((2, 3, 5)) > 0.25
    mask[0, 1] = False  # Exercise a fully masked row, including its derivative.
    output, cache = attention_forward(q, k, v, mask=mask, temperature=0.7)
    analytical = attention_backward(upstream, cache)
    errors = {}
    for name, array, grad in zip(("dQ", "dK", "dV"), (q, k, v), analytical):
        numeric = np.empty_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + epsilon
            plus, _ = attention_forward(q, k, v, mask=mask, temperature=0.7)
            array[index] = original - epsilon
            minus, _ = attention_forward(q, k, v, mask=mask, temperature=0.7)
            array[index] = original
            numeric[index] = np.sum((plus-minus)*upstream) / (2*epsilon)
        errors[name] = {"max_absolute_error": float(np.max(np.abs(numeric-grad))),
                        "relative_l2_error": float(np.linalg.norm(numeric-grad) /
                                                   max(np.linalg.norm(numeric)+np.linalg.norm(grad), 1e-12))}
        np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=2e-8)
    return {"seed": seed, "epsilon": epsilon, "dtype": "float64",
            "objective": "sum(attention(Q,K,V) * upstream)", "passed": True,
            "errors": errors}


def main():
    report = check()
    path = Path("docs/results/gradient_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
