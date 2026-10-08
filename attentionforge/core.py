"""Scaled dot-product attention and its analytical backward pass.

No autograd and no ML framework. True in a boolean mask means allowed.
Inputs have matching leading dimensions: (..., queries/keys, channels).
"""
from dataclasses import dataclass
import numpy as np


def _validate(q, k, v):
    if not all(isinstance(x, np.ndarray) for x in (q, k, v)):
        raise TypeError("q, k, v must be NumPy arrays")
    if any(x.ndim < 2 for x in (q, k, v)):
        raise ValueError("expected (..., sequence, channels) arrays")
    if not (q.shape[:-2] == k.shape[:-2] == v.shape[:-2]):
        raise ValueError("q, k, v must have matching leading dimensions")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("query/key channels and key/value lengths must match")
    if any(min(x.shape) <= 0 for x in (q, k, v)):
        raise ValueError("dimensions must be nonzero")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k, v must have the same dtype")
    if q.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("use float32 or float64 inputs")
    if not all(np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("q, k, v must contain finite values")


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is out of range")


def causal_mask(query_length, key_length=None, *, query_offset=0):
    """Allow key j when j <= query_offset + i (including cached decoding)."""
    key_length = query_length if key_length is None else key_length
    _positive_integer(query_length, "query_length")
    _positive_integer(key_length, "key_length")
    _positive_integer(query_offset, "query_offset", allow_zero=True)
    return np.arange(key_length)[None, :] <= (
        np.arange(query_length)[:, None] + query_offset
    )


def _mask(mask, shape, causal, offset):
    _positive_integer(offset, "query_offset", allow_zero=True)
    allowed = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("mask must be boolean; True means allowed")
        try:
            allowed = np.broadcast_to(mask, shape)
        except ValueError as exc:
            raise ValueError(f"mask cannot broadcast to {shape}") from exc
    if causal:
        triangle = causal_mask(shape[-2], shape[-1], query_offset=offset)
        allowed = triangle if allowed is None else allowed & triangle
    return allowed


def _scale(width, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    # A Python scalar preserves array dtype on both NumPy 1.x and 2.x.
    return float(1.0 / (np.sqrt(width) * temperature))


def stable_softmax(scores, mask=None):
    """Softmax over the last axis; fully masked rows are exactly zero.

    Subtracting the row maximum avoids overflow in exp. A boolean mask can
    broadcast to scores. Scores themselves must be finite floating-point values.
    """
    scores = np.asarray(scores)
    if scores.ndim == 0 or scores.shape[-1] == 0:
        raise ValueError("scores need a nonempty last dimension")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError("scores must be floating point")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("mask must be boolean")
        masked = np.where(np.broadcast_to(mask, scores.shape), scores, -np.inf)
    else:
        masked = scores
    maximum = np.max(masked, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    exp_scores = np.exp(masked - maximum)
    denominator = exp_scores.sum(axis=-1, keepdims=True)
    return np.divide(exp_scores, denominator, out=np.zeros_like(exp_scores),
                     where=denominator > 0)


@dataclass
class AttentionCache:
    """Arrays retained for backward. Do not mutate their inputs before backward."""
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    weights: np.ndarray
    scale: float


def attention_forward(q, k, v, *, mask=None, causal=False, query_offset=0,
                      temperature=1.0):
    """Return (output, cache), retaining probabilities for manual backward."""
    _validate(q, k, v)
    scale = _scale(q.shape[-1], temperature)
    scores = (q @ np.swapaxes(k, -2, -1)) * scale
    allowed = _mask(mask, scores.shape, causal, query_offset)
    weights = stable_softmax(scores, allowed)
    return weights @ v, AttentionCache(q, k, v, weights, scale)


def scaled_dot_product_attention(q, k, v, *, mask=None, causal=False,
                                 query_offset=0, temperature=1.0):
    """Return (output, weights). Accept 2D, batched, or multi-head inputs."""
    output, cache = attention_forward(q, k, v, mask=mask, causal=causal,
                                     query_offset=query_offset,
                                     temperature=temperature)
    return output, cache.weights


def attention_backward(d_output, cache):
    """Return dQ, dK, dV using the softmax Jacobian-vector product.

    dS = P * (dP - sum(dP * P)); no full Jacobian is ever constructed.
    """
    expected = cache.q.shape[:-1] + (cache.v.shape[-1],)
    if d_output.shape != expected:
        raise ValueError(f"d_output must have shape {expected}")
    if d_output.dtype != cache.q.dtype or not np.isfinite(d_output).all():
        raise ValueError("d_output must have the input dtype and finite values")
    p = cache.weights
    d_v = np.swapaxes(p, -2, -1) @ d_output
    d_p = d_output @ np.swapaxes(cache.v, -2, -1)
    d_s = p * (d_p - (d_p * p).sum(axis=-1, keepdims=True))
    d_q = (d_s @ cache.k) * cache.scale
    d_k = (np.swapaxes(d_s, -2, -1) @ cache.q) * cache.scale
    return d_q, d_k, d_v


def attention_entropy(weights):
    """Per-query entropy in nats; a fully masked row has entropy zero."""
    weights = np.asarray(weights)
    log_p = np.zeros_like(weights)
    np.log(weights, out=log_p, where=weights > 0)
    return -(weights * log_p).sum(axis=-1)
