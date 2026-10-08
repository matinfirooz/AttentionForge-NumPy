"""Evaluate the teaching checkpoint on a fresh synthetic test sample."""
import argparse
import json
from pathlib import Path
import numpy as np
from attentionforge import cross_entropy
from attentionforge.recall import RecallModel, make_recall_batch


def evaluate(*, checkpoint="docs/results/recall_model.npz", examples=4096,
             seed=2026, slots=6, output="docs/results/eval_report.json"):
    if examples <= 0:
        raise ValueError("examples must be positive")
    model = RecallModel.load(checkpoint)
    rng = np.random.default_rng(seed)
    correct, weighted_loss, count = 0, 0.0, 0
    while count < examples:
        n = min(256, examples-count)
        q, m, targets, _ = make_recall_batch(rng, n, slots=slots,
            d_model=model.config["d_model"], num_values=model.config["num_values"])
        logits, _, _ = model.forward(q, m)
        loss, _ = cross_entropy(logits, targets)
        correct += int(np.sum(logits.argmax(axis=-1) == targets))
        weighted_loss += loss*n
        count += n
    report = {"task": "synthetic associative recall", "checkpoint": str(checkpoint),
              "test_seed": seed, "examples": examples, "slots": slots,
              "accuracy": correct/examples, "correct": correct,
              "cross_entropy": weighted_loss/examples,
              "note": "Fresh random sample; this seed is separate from training and validation. Single checkpoint; not a real-world benchmark."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Fresh synthetic test: {correct}/{examples}, accuracy {correct/examples:.2%}, loss {weighted_loss/examples:.6f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="docs/results/recall_model.npz")
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--output", default="docs/results/eval_report.json")
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
