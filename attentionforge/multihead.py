"""Multi-head attention with manual projection gradients and a bounded KV cache."""
from dataclasses import dataclass
import numpy as np
from .core import attention_forward, attention_backward, _positive_integer
from .position import apply_rope


@dataclass
class MHACache:
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    attention: object
    joined: np.ndarray
    query_offset: int
    key_offset: int


class KVCache:
    """Preallocated projected keys/values for autoregressive self-attention.

    A cache is for a single sequence batch and a single layer/model. Reset it
    before another sequence or after changing model weights. append() rejects
    overflow before mutating. Existing keys are never projected or rotated again.
    """
    def __init__(self, batch_size, num_heads, max_length, head_dim, *, dtype=np.float64):
        for name, value in (("batch_size", batch_size), ("num_heads", num_heads),
                            ("max_length", max_length), ("head_dim", head_dim)):
            _positive_integer(value, name)
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("cache dtype must be float32 or float64")
        shape = (batch_size, num_heads, max_length, head_dim)
        self.k = np.empty(shape, dtype=dtype)
        self.v = np.empty(shape, dtype=dtype)
        self.length = 0

    @property
    def capacity(self):
        return self.k.shape[-2]

    @property
    def nbytes(self):
        return self.k.nbytes + self.v.nbytes

    def reset(self):
        self.length = 0

    def append(self, k, v):
        if k.ndim != 4 or v.shape != k.shape:
            raise ValueError("cache append expects matching (B,H,T,D) arrays")
        if (k.shape[:2] != self.k.shape[:2] or k.shape[-1] != self.k.shape[-1]
                or k.shape[-2] <= 0):
            raise ValueError("cache batch, head, width, or length mismatch")
        if k.dtype != self.k.dtype or v.dtype != self.v.dtype:
            raise TypeError("cache dtype mismatch")
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise ValueError("cache append needs finite values")
        end = self.length + k.shape[-2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded")
        self.k[..., self.length:end, :] = k
        self.v[..., self.length:end, :] = v
        self.length = end
        return self.k[..., :end, :], self.v[..., :end, :]


class MultiHeadAttention:
    """Bias-free, trainable self/cross attention. No residual, dropout, or FFN.

    forward returns (output, weights, cache_or_None). backward returns three
    input gradients and a parameter-gradient dict. For self-attention, add the
    three input gradients. Forward caches are caller-owned, so several can
    coexist. Weights must remain unchanged until their backward calls finish.
    """
    def __init__(self, d_model, num_heads, *, seed=42, dtype=np.float64, rope=False):
        _positive_integer(d_model, "d_model")
        _positive_integer(num_heads, "num_heads")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("use float32 or float64")
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.dtype, self.rope = np.dtype(dtype), rope
        if rope and self.head_dim % 2:
            raise ValueError("rotary attention requires an even head_dim")
        rng = np.random.default_rng(seed)
        self.params = {
            name: (rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)).astype(dtype)
            for name in ("Wq", "Wk", "Wv", "Wo")
        }

    def _input(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("inputs must have shape (batch, sequence, d_model)")
        if min(x.shape) <= 0:
            raise ValueError("input dimensions must be nonzero")
        if x.dtype != self.dtype:
            raise TypeError(f"input dtype must be {self.dtype}")
        if not np.isfinite(x).all():
            raise ValueError("inputs must be finite")

    def _split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _join(self, x):
        b, _, t, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, self.d_model)

    def forward(self, query, key=None, value=None, *, mask=None, causal=False,
                query_offset=0, key_offset=0, temperature=1.0, return_cache=False):
        key = query if key is None else key
        value = key if value is None else value
        for x in (query, key, value):
            self._input(x)
        if not (query.shape[0] == key.shape[0] == value.shape[0]):
            raise ValueError("batch sizes must match")
        _positive_integer(key_offset, "key_offset", allow_zero=True)
        if causal and key_offset:
            raise ValueError("causal masks use key positions starting at zero")
        q = self._split(query @ self.params["Wq"])
        k = self._split(key @ self.params["Wk"])
        v = self._split(value @ self.params["Wv"])
        if self.rope:
            q = apply_rope(q, offset=query_offset)
            k = apply_rope(k, offset=key_offset)
        out, attn = attention_forward(q, k, v, mask=mask, causal=causal,
                                      query_offset=query_offset, temperature=temperature)
        joined = self._join(out)
        cache = MHACache(query, key, value, attn, joined, query_offset, key_offset) if return_cache else None
        return joined @ self.params["Wo"], attn.weights, cache

    def backward(self, d_output, cache):
        if not isinstance(cache, MHACache):
            raise TypeError("pass a cache from forward(return_cache=True)")
        if d_output.shape != cache.joined.shape or d_output.dtype != self.dtype:
            raise ValueError("d_output must match forward output shape and dtype")
        if not np.isfinite(d_output).all():
            raise ValueError("d_output must be finite")
        flat = lambda x: x.reshape(-1, self.d_model)
        grads = {"Wo": flat(cache.joined).T @ flat(d_output)}
        d_joined = d_output @ self.params["Wo"].T
        d_q, d_k, d_v = attention_backward(self._split(d_joined), cache.attention)
        if self.rope:
            d_q = apply_rope(d_q, offset=cache.query_offset, inverse=True)
            d_k = apply_rope(d_k, offset=cache.key_offset, inverse=True)
        d_q, d_k, d_v = (self._join(x) for x in (d_q, d_k, d_v))
        inputs = (cache.query, cache.key, cache.value)
        derivatives = (d_q, d_k, d_v)
        for name, x, dx in zip(("Wq", "Wk", "Wv"), inputs, derivatives):
            grads[name] = flat(x).T @ flat(dx)
        d_inputs = tuple(dx @ self.params[name].T for dx, name in
                         zip(derivatives, ("Wq", "Wk", "Wv")))
        return (*d_inputs, grads)

    def decode(self, x_new, cache, *, temperature=1.0):
        """Append one or more new tokens and attend causally to the prefix."""
        self._input(x_new)
        if not isinstance(cache, KVCache):
            raise TypeError("cache must be KVCache")
        expected = (x_new.shape[0], self.num_heads, self.head_dim)
        if (cache.k.shape[0], cache.k.shape[1], cache.k.shape[-1]) != expected:
            raise ValueError("cache does not match this model/batch")
        if cache.k.dtype != self.dtype:
            raise TypeError("cache dtype does not match model")
        # Validate before append so invalid temperature never changes the cache.
        from .core import _scale
        _scale(self.head_dim, temperature)
        offset = cache.length
        q = self._split(x_new @ self.params["Wq"])
        k = self._split(x_new @ self.params["Wk"])
        v = self._split(x_new @ self.params["Wv"])
        if self.rope:
            q, k = apply_rope(q, offset=offset), apply_rope(k, offset=offset)
        k_all, v_all = cache.append(k, v)
        out, attn = attention_forward(q, k_all, v_all, causal=True,
                                      query_offset=offset, temperature=temperature)
        return self._join(out) @ self.params["Wo"], attn.weights
