"""Small analytical-gradient utilities: Adam, cross entropy, clipping."""
import numpy as np
from .core import stable_softmax


def cross_entropy(logits, targets):
    """Mean categorical loss and its gradient; logits=(B,C), targets=(B,)."""
    if logits.ndim != 2 or logits.shape[0] == 0 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits (B,C), targets (B,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("targets must be integers")
    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("targets out of range")
    p = stable_softmax(logits)
    maximum = logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(logits - maximum).sum(axis=-1)) + maximum[:, 0]
    loss = np.mean(log_z - logits[np.arange(len(targets)), targets])
    d_logits = p.copy()
    d_logits[np.arange(len(targets)), targets] -= 1
    return float(loss), d_logits / len(targets)


def clip_gradients(grads, max_norm=1.0):
    """In-place global L2 clipping; return the norm before clipping."""
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be positive and finite")
    if not all(np.isfinite(g).all() for g in grads.values()):
        raise ValueError("gradients must be finite")
    norm = float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values())))
    factor = min(1.0, max_norm / (norm + 1e-12))
    for g in grads.values():
        g *= factor
    return norm


class Adam:
    def __init__(self, params, *, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("lr and eps must be positive and finite")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0,1)")
        self.params, self.lr = params, lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {name: np.zeros_like(x) for name, x in params.items()}
        self.v = {name: np.zeros_like(x) for name, x in params.items()}
        self.step_count = 0

    def step(self, grads):
        if grads.keys() != self.params.keys():
            raise ValueError("gradient keys must match parameters exactly")
        for name, g in grads.items():
            if g.shape != self.params[name].shape or g.dtype != self.params[name].dtype:
                raise ValueError(f"gradient shape or dtype mismatch for {name}")
            if not np.isfinite(g).all():
                raise ValueError(f"nonfinite gradient for {name}")
        self.step_count += 1
        for name, x in self.params.items():
            g = grads[name]
            self.m[name] *= self.beta1
            self.m[name] += (1 - self.beta1) * g
            self.v[name] *= self.beta2
            self.v[name] += (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[name] / (1 - self.beta2 ** self.step_count)
            x -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
