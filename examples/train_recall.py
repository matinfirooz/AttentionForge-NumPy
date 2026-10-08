"""Learn associative recall with no autograd; emit measured results and SVGs."""
import argparse
import csv
import json
import platform
from pathlib import Path
import time
import numpy as np
from attentionforge import Adam, cross_entropy, clip_gradients, attention_entropy
from attentionforge.recall import RecallModel, make_recall_batch
from attentionforge.visuals import training_plot, recall_heads


def train(*, steps=600, batch_size=64, seed=42, output="docs/results"):
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1000)
    validation = make_recall_batch(val_rng, 1024)
    query, memory, targets, info = validation
    model = RecallModel(seed=seed)
    optimizer = Adam(model.params, lr=0.003)
    before_logits, before_weights, _ = model.forward(query[:1], memory[:1])
    initial_weights = before_weights.copy()
    history = []
    started = time.perf_counter()

    def evaluate(step, train_loss=None, train_accuracy=None):
        logits, weights, _ = model.forward(query, memory)
        loss, _ = cross_entropy(logits, targets)
        record = {"step": step, "train_loss": train_loss,
                  "train_accuracy": train_accuracy,
                  "val_loss": loss,
                  "val_accuracy": float(np.mean(logits.argmax(axis=-1) == targets)),
                  "mean_head_entropy": float(attention_entropy(weights).mean())}
        history.append(record)
        print(f"step {step:4d} | val loss {loss:.4f} | val accuracy {record['val_accuracy']:.2%}", flush=True)

    evaluate(0)
    for step in range(1, steps + 1):
        tq, tm, tt, _ = make_recall_batch(rng, batch_size)
        loss, accuracy, grads, _ = model.loss_and_grads(tq, tm, tt)
        clip_gradients(grads, 1.0)
        optimizer.step(grads)
        if step % 25 == 0 or step == steps:
            evaluate(step, loss, accuracy)
    elapsed = time.perf_counter() - started
    logits, final_weights, _ = model.forward(query[:1], memory[:1])
    model.save(destination / "recall_model.npz")
    restored = RecallModel.load(destination / "recall_model.npz")
    restored_logits, _, _ = restored.forward(query[:1], memory[:1])
    np.testing.assert_array_equal(logits, restored_logits)
    training_plot(history, destination.parent / "assets" / "training.svg")
    recall_heads(initial_weights, final_weights, info, int(logits.argmax()),
                 destination.parent / "assets" / "recall_heads.svg")
    with (destination / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {"task": "synthetic associative recall", "seed": seed,
              "steps": steps, "batch_size": batch_size, "slots": 6,
              "num_keys": 16, "num_values": 8, "validation_examples": 1024,
              "validation_seed": seed + 1000,
              "parameters": sum(p.size for p in model.params.values()),
              "initial": history[0], "final": history[-1],
              "elapsed_seconds": elapsed, "python": platform.python_version(),
              "numpy": np.__version__, "history": history,
              "checkpoint_reload_exact": True,
              "note": "Fresh random training batches; fixed validation used only for monitoring. Single-seed teaching experiment."}
    (destination / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}; {report['parameters']:,} parameters; {elapsed:.2f} seconds.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="docs/results")
    args = parser.parse_args()
    train(steps=args.steps, batch_size=args.batch_size, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
