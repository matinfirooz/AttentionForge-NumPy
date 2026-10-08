"""Run tests, training, checks, benchmarks, and graphics from one command."""
import json
import os
from pathlib import Path
import platform
import sys
import unittest


def main():
    # Set before importing NumPy. Existing explicit thread settings are honored.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    from examples.train_recall import train
    from examples.evaluate_recall import evaluate
    from examples.gradient_check import main as gradients
    from examples.decode import main as decode
    from examples.benchmark import benchmark
    from examples.build_showcase import main as graphics
    from examples.build_notebook import main as notebook
    train()
    evaluate()
    gradients()
    decode()
    benchmark()
    graphics()
    notebook()
    report = {"tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "successful": result.wasSuccessful(), "python": platform.python_version(),
              "command": "python -m examples.reproduce"}
    (root/"docs/results/verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("\nReproduction complete. Open docs/attention_explorer.html and README.md.")


if __name__ == "__main__":
    main()
