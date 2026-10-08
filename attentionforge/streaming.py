"""Tiled online-softmax attention: no full score/probability matrix."""
import numpy as np
from .core import _validate, _positive_integer, _scale


def streaming_attention(q, k, v, *, query_block=32, key_block=64, mask=None,
                        causal=False, query_offset=0, temperature=1.0):
    """Exact attention up to floating-point order; return output only.

    Running maximum m, exponential sum l, and unnormalized weighted sum a
    are rescaled when a new key tile raises m. Explicit masks are broadcast
    views; causal masks are generated per tile. This is a readable CPU NumPy
    algorithm, not a FlashAttention GPU kernel. This function has no backward.
    """
    _validate(q, k, v)
    _positive_integer(query_block, "query_block")
    _positive_integer(key_block, "key_block")
    _positive_integer(query_offset, "query_offset", allow_zero=True)
    scale = _scale(q.shape[-1], temperature)
    leading, nq, nk = q.shape[:-2], q.shape[-2], k.shape[-2]
    shape = leading + (nq, nk)
    allowed = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("mask must be boolean; True means allowed")
        allowed = np.broadcast_to(mask, shape)
    result = np.empty(leading + (nq, v.shape[-1]), dtype=q.dtype)
    for qi in range(0, nq, query_block):
        qe = min(qi + query_block, nq)
        qs = q[..., qi:qe, :]
        m = np.full(leading + (qe - qi, 1), -np.inf, dtype=q.dtype)
        normalizer = np.zeros_like(m)
        accumulator = np.zeros(leading + (qe - qi, v.shape[-1]), dtype=q.dtype)
        for ki in range(0, nk, key_block):
            ke = min(ki + key_block, nk)
            scores = (qs @ np.swapaxes(k[..., ki:ke, :], -2, -1)) * scale
            tile_mask = None if allowed is None else allowed[..., qi:qe, ki:ke]
            if causal:
                triangle = np.arange(ki, ke)[None, :] <= (
                    np.arange(qi, qe)[:, None] + query_offset
                )
                tile_mask = triangle if tile_mask is None else tile_mask & triangle
            if tile_mask is not None:
                scores = np.where(tile_mask, scores, -np.inf)
            new_m = np.maximum(m, scores.max(axis=-1, keepdims=True))
            # A tile can be entirely masked; use zero only as an exp reference.
            reference = np.where(np.isfinite(new_m), new_m, 0)
            correction = np.exp(m - reference)
            probabilities = np.exp(scores - reference)
            normalizer = correction * normalizer + probabilities.sum(axis=-1, keepdims=True)
            accumulator = correction * accumulator + probabilities @ v[..., ki:ke, :]
            m = new_m
        result[..., qi:qe, :] = np.divide(
            accumulator, normalizer, out=np.zeros_like(accumulator),
            where=normalizer > 0,
        )
    return result


def attention_storage_elements(query_length, key_length, value_width, *,
                               query_block=32, key_block=64, batch_heads=1):
    """Compare named state sizes, NOT measured process peak memory.

    dense_score_and_weights: two complete score-shaped arrays.
    tiled_score_and_exp: two score tiles across all batch/head dimensions.
    tiled_running_state: m, l, a for one query tile.
    BLAS workspaces, temporary expressions, inputs, mask, and output excluded.
    """
    for name, value in (("query_length", query_length), ("key_length", key_length),
                        ("value_width", value_width), ("query_block", query_block),
                        ("key_block", key_block), ("batch_heads", batch_heads)):
        _positive_integer(value, name)
    qb, kb = min(query_block, query_length), min(key_block, key_length)
    return {
        "dense_score_and_weights": 2 * batch_heads * query_length * key_length,
        "tiled_score_and_exp": 2 * batch_heads * qb * kb,
        "tiled_running_state": batch_heads * qb * (value_width + 2),
        "output": batch_heads * query_length * value_width,
    }
