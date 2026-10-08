"""Create and execute a notebook with stdlib JSON and NumPy only.

The tutorial contains predetermined code; its captured outputs come from
executing those cells. A notebook viewer/kernel is optional for exploration.
"""
import contextlib
import io
import json
from pathlib import Path
import platform


CELLS = [
    ("markdown", """# AttentionForge: attention with nothing hidden

**Created for [@matinfirooz](https://github.com/matinfirooz).**

This executed tutorial goes from a three-query numerical example to manual
gradients, multi-head attention, rotary positions, tiled online softmax, cached
decoding, and a trained retrieval model. All tensor math uses NumPy.

The saved outputs are actual runs. To explore, open this notebook in a
Jupyter-compatible editor with a Python kernel and NumPy installed.
"""),
    ("code", """from pathlib import Path
import sys, json
root = Path.cwd()
if not (root / "attentionforge").is_dir():
    root = root.parent
if not (root / "attentionforge").is_dir():
    raise RuntimeError("Open this notebook from the repository or notebooks directory")
sys.path.insert(0, str(root))
import numpy as np
from attentionforge import (
    scaled_dot_product_attention, attention_forward, attention_backward,
    MultiHeadAttention, KVCache, streaming_attention, apply_rope,
    sinusoidal_encoding, attention_entropy
)
np.set_printoptions(precision=4, suppress=True)
print("NumPy:", np.__version__)
"""),
    ("markdown", r"""## 1. A query compares keys and retrieves values

Queries and keys share a channel width. Values can have a different width.

$$S=QK^T/\sqrt{d_k},\qquad P=\operatorname{softmax}(S),\qquad O=PV.$$
"""),
    ("code", """q = np.array([[2., 0.], [0., 2.], [1., 1.]])
k = np.array([[1., 0.], [0., 1.], [1., 1.]])
v = np.array([[10., 0.], [0., 20.], [30., 30.]])
scores = q @ k.T / np.sqrt(q.shape[-1])
output, weights = scaled_dot_product_attention(q, k, v)
print("Scaled scores:\\n", scores)
print("Probabilities:\\n", weights)
print("Output:\\n", output)
np.testing.assert_allclose(weights.sum(axis=-1), 1)
"""),
    ("markdown", """## 2. Visibility changes what a query can retrieve

A causal query only sees its prefix. Boolean masks use **True = allowed**.
A fully masked row is defined to return zeros. Temperature controls the
sharpness of visible probabilities.
"""),
    ("code", """causal_output, causal_weights = scaled_dot_product_attention(q, k, v, causal=True)
print("Causal probabilities:\\n", causal_weights)
mask = np.ones((3, 3), dtype=bool)
mask[1] = False
masked_output, masked_weights = scaled_dot_product_attention(q, k, v, mask=mask)
print("Fully masked query output:", masked_output[1])
for temperature in (0.25, 1., 2.):
    _, p = scaled_dot_product_attention(q, k, v, temperature=temperature)
    print("Temperature", temperature, "first-row entropy:", round(float(attention_entropy(p)[0]), 4))
"""),
    ("markdown", r"""## 3. Differentiate without autograd

The softmax Jacobian-vector product is:

$$D_S=P\odot\big(D_P-\operatorname{rowsum}(D_P\odot P)\big).$$

The checker independently perturbs every Q/K/V element. Its objective is the
scalar inner product of the attention output and a fixed upstream tensor.
"""),
    ("code", """output, saved = attention_forward(q, k, v)
dq, dk, dv = attention_backward(np.ones_like(output), saved)
print("dQ for sum(output):\\n", dq)
print("dK for sum(output):\\n", dk)
print("dV for sum(output):\\n", dv)
from examples.gradient_check import check
print(json.dumps(check(), indent=2))
"""),
    ("markdown", """## 4. Project, split heads, attend, join, project again

The model width is split equally across heads. In cross-attention, queries can
have a different sequence length from the memory. The backward returns the
three input branches separately; add branches when their input is shared.
"""),
    ("code", """rng = np.random.default_rng(42)
x = rng.normal(size=(2, 7, 32))
memory = rng.normal(size=(2, 11, 32))
layer = MultiHeadAttention(32, 4, seed=42)
y, p, saved_mha = layer.forward(x, memory, memory, return_cache=True)
dx, dkey, dvalue, grads = layer.backward(np.ones_like(y), saved_mha)
print("Output:", y.shape, "head weights:", p.shape)
print("Input gradient shapes:", dx.shape, dkey.shape, dvalue.shape)
print("Parameter gradients:", {name: grad.shape for name, grad in grads.items()})
"""),
    ("markdown", """## 5. Rotary positions preserve norms

Each adjacent channel pair rotates by an angle determined by its absolute
position and frequency. A common position shift leaves Q/K dot products
unchanged. This implementation uses adjacent pairs, not split-half pairs.
"""),
    ("code", """r = rng.normal(size=(5, 8))
rotated = apply_rope(r, offset=12)
print("Norm preservation max error:", float(np.max(np.abs(np.linalg.norm(r, axis=-1)-np.linalg.norm(rotated, axis=-1)))))
np.testing.assert_allclose(apply_rope(rotated, offset=12, inverse=True), r, atol=1e-12)
print("Sinusoidal encoding, first three positions:\\n", sinusoidal_encoding(3, 8))
"""),
    ("markdown", """## 6. Stream tiles through an online softmax

Running maximum, normalizer, and weighted value accumulator replace the full
score/probability matrices. Exactness here means agreement up to floating-point
evaluation order. The tiled path is forward-only and is an educational CPU
implementation, not a GPU kernel.
"""),
    ("code", """qs = rng.normal(size=(2, 4, 17, 8))
ks = rng.normal(size=(2, 4, 23, 8))
vs = rng.normal(size=(2, 4, 23, 5))
dense, _ = scaled_dot_product_attention(qs, ks, vs, causal=True)
tiled = streaming_attention(qs, ks, vs, causal=True, query_block=5, key_block=7)
np.testing.assert_allclose(tiled, dense, rtol=1e-12, atol=1e-12)
print("Dense/tiled max absolute error:", float(np.max(np.abs(tiled-dense))))
"""),
    ("markdown", """## 7. Cached decoding must equal full causal attention

The cache stores projected keys and values. New queries use their absolute
position; rotary keys are rotated only once, at insertion. A chunk still hides
its own future tokens. Reset the cache for a new sequence or changed weights.
"""),
    ("code", """layer = MultiHeadAttention(32, 4, seed=42, rope=True)
full, _, _ = layer.forward(x, causal=True)
cache = KVCache(2, 4, max_length=7, head_dim=8)
chunks, offset = [], 0
for size in (2, 1, 4):
    out, _ = layer.decode(x[:, offset:offset+size], cache)
    chunks.append(out)
    offset += size
decoded = np.concatenate(chunks, axis=1)
np.testing.assert_allclose(decoded, full, rtol=1e-12, atol=1e-12)
print("Full/chunked max absolute error:", float(np.max(np.abs(decoded-full))))
print("Cache tokens:", cache.length, "preallocated bytes:", cache.nbytes)
"""),
    ("markdown", """## 8. Inspect a memory model trained with the same analytical gradients

The task redraws random values and unique keys; a queried key must recover its
current value. The included checkpoint has 4,360 parameters. It is a synthetic
retrieval model, not a pretrained language model.

Run `python -m examples.train_recall` from the repository root to retrain it.
Run `python -m examples.evaluate_recall` for a separate fresh sample.
"""),
    ("code", """from attentionforge.recall import RecallModel, make_recall_batch
trained = RecallModel.load(root / "docs/results/recall_model.npz")
tq, tm, targets, info = make_recall_batch(np.random.default_rng(2026), 1)
logits, heads, _ = trained.forward(tq, tm)
chosen = int(info["chosen"][0])
print("Memory keys:", info["keys"][0])
print("Memory values:", info["values"][0])
print("Query key:", info["keys"][0, chosen])
print("Target:", targets[0], "prediction:", logits.argmax(axis=-1)[0])
print("Each head's retrieval probabilities:\\n", heads[0, :, 0, :])
report = json.loads((root / "docs/results/train_report.json").read_text())
print("Measured validation accuracy:", report["final"]["val_accuracy"])
"""),
    ("markdown", """## Continue exploring

- [Complete math and derivative guide](../docs/MATH.md)
- [Training curves](../docs/assets/training.svg)
- [Attention heads before/after learning](../docs/assets/recall_heads.svg)
- [Raw benchmark report](../docs/results/benchmark.json)
- [Offline interactive explorer](../docs/attention_explorer.html)

The browser explorer selects precomputed NumPy tensors. Download/open its HTML
file locally to interact with it. All runtime examples need only NumPy and the
standard library; notebook editors are optional.
"""),
]


def main():
    namespace = {"__name__": "__notebook__"}
    cells, execution = [], 0
    for cell_type, text in CELLS:
        cell = {"cell_type": cell_type, "metadata": {}, "source": text.splitlines(keepends=True)}
        if cell_type == "code":
            execution += 1
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(text, f"notebook_cell_{execution}", "exec"), namespace)
            captured = stream.getvalue()
            cell.update(execution_count=execution, outputs=[{
                "output_type": "stream", "name": "stdout", "text": captured.splitlines(keepends=True)
            }] if captured else [])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4, "nbformat_minor": 4}
    path = Path("notebooks/Attention_From_Scratch.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1)+"\n", encoding="utf-8")
    print(f"Wrote {path}: {len(cells)} cells, {execution} executed code cells.")


if __name__ == "__main__":
    main()
