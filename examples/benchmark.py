"""Reproducible dense/tiled comparison; timings and storage estimates separate."""
import argparse
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
from attentionforge import scaled_dot_product_attention, streaming_attention, attention_storage_elements
from attentionforge.visuals import SVG, FG, MUTED, CYAN, GOLD


def median_time(function, repeats):
    function()  # Warm up BLAS and allocator; do not include this in timing.
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter()-started)
    return float(np.median(samples))


def benchmark(*, lengths=(64, 128, 256, 512, 1024), repeats=5,
              output="docs/results/benchmark.json"):
    if repeats <= 0 or any(n <= 0 for n in lengths):
        raise ValueError("lengths and repeats must be positive")
    rng = np.random.default_rng(42)
    records = []
    for length in lengths:
        q, k, v = [rng.normal(size=(1, 4, length, 32)) for _ in range(3)]
        dense = lambda: scaled_dot_product_attention(q, k, v, causal=True)[0]
        tiled = lambda: streaming_attention(q, k, v, causal=True, query_block=32, key_block=64)
        d, t = dense(), tiled()
        error = float(np.max(np.abs(d-t)))
        np.testing.assert_allclose(t, d, rtol=1e-11, atol=1e-12)
        state = attention_storage_elements(length, length, 32, query_block=32, key_block=64, batch_heads=4)
        record = {"length": length, "dense_ms": median_time(dense, repeats)*1000,
                  "tiled_ms": median_time(tiled, repeats)*1000,
                  "max_absolute_error": error,
                  "named_storage_bytes": {name: count*q.dtype.itemsize for name, count in state.items()}}
        records.append(record)
        print(f"N={length:4d} | dense {record['dense_ms']:8.3f} ms | tiled {record['tiled_ms']:8.3f} ms | max error {error:.2e}", flush=True)
    report = {"seed": 42, "dtype": "float64", "batch": 1, "heads": 4,
              "head_dim": 32, "query_block": 32, "key_block": 64, "causal": True,
              "repeats": repeats, "statistic": "median after one warmup",
              "python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(),
              "blas_thread_environment": {name: os.environ.get(name) for name in
                                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "records": records,
              "storage_note": "Named score/exp/running-state arrays only. Not measured peak RSS. Inputs, output, masks, expression temporaries and BLAS workspace excluded. Tiled stores no full N x N scores/probabilities.",
              "timing_note": "Readable Python loops can make tiled attention slower. Timings are local CPU measurements, not a GPU or general speedup claim."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    memory_figure(records, path.parent.parent / "assets" / "storage.svg")
    return report


def memory_figure(records, path):
    svg = SVG(1100, 420, "Theoretical named state storage for dense and tiled attention")
    svg.text(35, 42, "Keep tiles. Skip the full attention matrix.", 25, FG, 700)
    svg.text(35, 71, "Named array storage, float64 · B=1 · H=4 · query tile=32 · key tile=64", 14, MUTED)
    top, bottom, x0, width = 130, 328, 110, 940
    maximum = max(r["named_storage_bytes"]["dense_score_and_weights"] for r in records)
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        yy = bottom - (bottom-top)*fraction
        svg.line(x0, yy, x0+width, yy, "#22344a")
        svg.text(x0-12, yy+4, f"{maximum*fraction/2**20:.0f} MiB", 12, MUTED, anchor="end")
    for i, r in enumerate(records):
        x = x0 + (i+0.5)*width/len(records)
        state = r["named_storage_bytes"]
        for shift, name, fill in ((-31, "dense_score_and_weights", GOLD),
                                  (7, "tiled_score_and_exp", CYAN)):
            height = max(2, (bottom-top)*state[name]/maximum)
            svg.rect(x+shift, bottom-height, 31, height, fill, 3)
        svg.text(x, 353, r["length"], 12, MUTED, anchor="middle")
    svg.rect(635, 94, 12, 12, GOLD, 2)
    svg.text(653, 105, "Dense scores + probabilities", 12, MUTED)
    svg.rect(895, 94, 12, 12, CYAN, 2)
    svg.text(913, 105, "Tiled scores + exp", 12, MUTED)
    svg.text(35, 389, "Excludes inputs, outputs, running state, masks, temporaries, and BLAS workspace. Not peak RAM.", 13, MUTED)
    svg.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="docs/results/benchmark.json")
    args = parser.parse_args()
    benchmark(lengths=args.lengths, repeats=args.repeats, output=args.output)


if __name__ == "__main__":
    main()
