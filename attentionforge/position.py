"""Sinusoidal positions and adjacent-pair rotary embeddings."""
import numpy as np
from .core import _positive_integer


def sinusoidal_encoding(length, width, *, offset=0, base=10000.0,
                        dtype=np.float64):
    _positive_integer(length, "length")
    _positive_integer(width, "width")
    _positive_integer(offset, "offset", allow_zero=True)
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one")
    positions = np.arange(offset, offset + length)[:, None]
    frequencies = base ** (-np.arange(0, width, 2) / width)
    angles = positions * frequencies[None, :]
    result = np.empty((length, width), dtype=dtype)
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles[:, :width // 2])
    return result


def apply_rope(x, *, offset=0, base=10000.0, inverse=False):
    """Rotate adjacent channel pairs; x has shape (..., sequence, even width).

    This uses interleaved pairs (0,1), (2,3), ... rather than a split-half
    checkpoint convention. Keys in a KV cache are rotated only at insertion.
    inverse=True applies the transpose rotation for the analytical backward.
    """
    x = np.asarray(x)
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("RoPE needs a nonzero, even channel dimension")
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("RoPE needs float32 or float64 inputs")
    _positive_integer(offset, "offset", allow_zero=True)
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one")
    frequencies = base ** (-np.arange(0, x.shape[-1], 2) / x.shape[-1])
    angles = np.arange(offset, offset + x.shape[-2])[:, None] * frequencies
    if inverse:
        angles = -angles
    c, s = np.cos(angles).astype(x.dtype), np.sin(angles).astype(x.dtype)
    result = np.empty_like(x)
    result[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    result[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return result
