"""A small trainable associative-memory task, not a pretrained language model."""
from pathlib import Path
import json
import numpy as np
from .multihead import MultiHeadAttention
from .optim import cross_entropy


def make_recall_batch(rng, batch_size=64, *, slots=6, num_keys=16,
                      num_values=8, d_model=32):
    """Sample unique keys, random values, and a query for one memory slot.

    Features are one-hot [key | value | zero padding]. Values are regenerated
    every batch, so a key has no permanent label. Slot order is randomized.
    """
    if not 1 <= slots <= num_keys or d_model < num_keys + num_values:
        raise ValueError("slots/feature dimensions are incompatible")
    keys = np.argsort(rng.random((batch_size, num_keys)), axis=1)[:, :slots]
    values = rng.integers(0, num_values, (batch_size, slots))
    chosen = rng.integers(0, slots, batch_size)
    memory = np.zeros((batch_size, slots, d_model), dtype=np.float64)
    rows, cols = np.arange(batch_size)[:, None], np.arange(slots)[None, :]
    memory[rows, cols, keys] = 1
    memory[rows, cols, num_keys + values] = 1
    query = np.zeros((batch_size, 1, d_model), dtype=np.float64)
    query[np.arange(batch_size), 0, keys[np.arange(batch_size), chosen]] = 1
    targets = values[np.arange(batch_size), chosen]
    return query, memory, targets, {"keys": keys, "values": values, "chosen": chosen}


class RecallModel:
    def __init__(self, *, d_model=32, num_heads=4, num_values=8, seed=42):
        self.config = {"d_model": d_model, "num_heads": num_heads,
                       "num_values": num_values, "seed": seed}
        self.attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        rng = np.random.default_rng(seed + 1)
        self.params = {**self.attention.params,
                       "Wc": rng.standard_normal((d_model, num_values)) / np.sqrt(d_model),
                       "bc": np.zeros(num_values, dtype=np.float64)}

    def forward(self, query, memory, *, training=False):
        output, weights, cache = self.attention.forward(query, memory, memory,
                                                       return_cache=training)
        hidden = output[:, 0, :]
        logits = hidden @ self.params["Wc"] + self.params["bc"]
        return logits, weights, (hidden, cache)

    def loss_and_grads(self, query, memory, targets):
        logits, weights, (hidden, cache) = self.forward(query, memory, training=True)
        loss, d_logits = cross_entropy(logits, targets)
        grads = {"Wc": hidden.T @ d_logits, "bc": d_logits.sum(axis=0)}
        d_hidden = d_logits @ self.params["Wc"].T
        _, _, _, attention_grads = self.attention.backward(d_hidden[:, None, :], cache)
        grads.update(attention_grads)
        accuracy = float(np.mean(logits.argmax(axis=-1) == targets))
        return loss, accuracy, grads, weights

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A scalar JSON string is safe to load with allow_pickle=False.
        np.savez_compressed(path, **self.params,
                            config=np.array(json.dumps(self.config)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(**json.loads(str(data["config"])))
            for name, parameter in model.params.items():
                saved = data[name]
                if saved.shape != parameter.shape or saved.dtype != parameter.dtype:
                    raise ValueError(f"checkpoint mismatch for {name}")
                if not np.isfinite(saved).all():
                    raise ValueError("checkpoint contains nonfinite parameters")
                parameter[...] = saved
        return model
