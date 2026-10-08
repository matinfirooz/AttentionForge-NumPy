"""AttentionForge: attention you can read, differentiate, train, and inspect."""
from .core import (AttentionCache, attention_forward, attention_backward,
                   scaled_dot_product_attention, stable_softmax, causal_mask,
                   attention_entropy)
from .multihead import MultiHeadAttention, KVCache
from .position import apply_rope, sinusoidal_encoding
from .streaming import streaming_attention, attention_storage_elements
from .optim import Adam, cross_entropy, clip_gradients

__version__ = "1.0.0"
__all__ = ["AttentionCache", "attention_forward", "attention_backward",
           "scaled_dot_product_attention", "stable_softmax", "causal_mask",
           "attention_entropy", "MultiHeadAttention", "KVCache", "apply_rope",
           "sinusoidal_encoding", "streaming_attention", "attention_storage_elements",
           "Adam", "cross_entropy", "clip_gradients"]
