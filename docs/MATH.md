# Attention, from the equation to its derivative

Use [`examples/walkthrough.py`](../examples/walkthrough.py) to print the actual
numbers. All formulas below use GitHub-compatible math delimiters.

## 1. Queries, keys, and values

A query asks what to retrieve. A key describes a memory entry. A value carries
the information to retrieve. Self-attention obtains all three from the same
sequence; cross-attention uses one sequence for queries and another for memory.

$$
Q = X_q W_Q,\qquad K = X_k W_K,\qquad V = X_v W_V.
$$

The bare attention function also accepts already projected arrays. For one
head, $Q\in\mathbb{R}^{L_q\times d_k}$,
$K\in\mathbb{R}^{L_k\times d_k}$, and
$V\in\mathbb{R}^{L_k\times d_v}$. Query and key lengths can differ.

## 2. Compare and scale

$$
S = \frac{QK^T}{\sqrt{d_k}\,\tau}.
$$

The dot product measures similarity. Scaling by $\sqrt{d_k}$ controls the
magnitude as the channel width increases. The optional temperature $\tau>0$
sharpens attention when lowered and softens it when raised. Standard attention
uses $\tau=1$.

The walkthrough uses:

$$
Q=\begin{bmatrix}2&0\\0&2\\1&1\end{bmatrix},\quad
K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}10&0\\0&20\\30&30\end{bmatrix}.
$$

For the first query, scores are approximately
$[1.4142,0,1.4142]$. The probabilities are approximately
$[0.4458,0.1084,0.4458]$, giving an output of approximately
$[17.8323,15.5419]$. These probabilities describe the supplied vectors;
they do not establish any language understanding.

## 3. Mask and normalize

A mask has **True for allowed entries**. A hidden score behaves as
$-\infty$, so its exponential is zero. Each nonempty row uses:

$$
m_i=\max_j S_{ij},\qquad
P_{ij}=\frac{\exp(S_{ij}-m_i)}{\sum_t\exp(S_{it}-m_i)}.
$$

Subtracting $m_i$ leaves the probabilities unchanged while keeping every
visible exponent nonpositive. The implementation defines a fully masked row
to have zero probabilities and zero output. That row has zero gradients too;
there is no division by zero.

For causal attention, query $i$ can use key $j$ only if
$j\leq i+\mathrm{query\_offset}$. An offset is essential during cached decoding:
a new query at absolute position 10 must be able to see keys 0 through 10.

## 4. Retrieve

$$
O=PV.
$$

Every output row is a weighted sum of value rows. For a nonempty mask, weights
are nonnegative and sum to one. A fully masked row is the defined zero case.

## 5. Split into heads

With $H$ heads and $d_h=d_{model}/H$, the implementation splits projected
tensors from $(B,L,d_{model})$ into $(B,H,L,d_h)$. Each head computes its own
attention. Concatenated head outputs are projected by $W_O$:

$$
Y=\operatorname{Concat}(O_1,\ldots,O_H)W_O.
$$

This repository's attention layer has no projection biases. The recall
classifier has its own output bias. Residual paths, feed-forward layers,
layer normalization, and dropout are outside this layer's scope.

## 6. Backward without autograd

Let $G=\partial\mathcal{L}/\partial O$ and
$c=1/(\sqrt{d_k}\,\tau)$. Matrix multiplication gives:

$$
\frac{\partial\mathcal{L}}{\partial V}=P^T G,\qquad
\frac{\partial\mathcal{L}}{\partial P}=GV^T.
$$

Write the second quantity as $D_P$. Applying the softmax Jacobian to each row
can be expressed without constructing an $L_k\times L_k$ Jacobian:

$$
(D_S)_{ij}=P_{ij}\left((D_P)_{ij}-\sum_t P_{it}(D_P)_{it}\right).
$$

Then:

$$
D_Q=cD_SK,\qquad D_K=cD_S^TQ.
$$

Because masked probabilities are zero, the same expression also zeros their
score gradients. For the output projection:

$$
D_{W_O}=O_{concat}^T D_Y,\qquad D_{O_{concat}}=D_YW_O^T.
$$

Flatten batch and sequence dimensions when computing projection parameter
gradients. For example:

$$
D_{W_Q}=X_q^TD_Q,\qquad D_{X_q}=D_QW_Q^T.
$$

When self-attention shares one input among Q, K, V, add the three input
gradients. For cross-attention with shared key/value memory, add the key and
value input gradients. This package returns the branches separately so both
uses are explicit.

The numerical checker perturbs each input element and estimates:

$$
\frac{\partial\mathcal{L}}{\partial x_i}\approx
\frac{\mathcal{L}(x_i+\epsilon)-\mathcal{L}(x_i-\epsilon)}{2\epsilon}.
$$

Tests also check all four projection matrices, the classifier, shared-input
gradients, and rotary gradients against finite differences.

## 7. Rotary positions

Adjacent channels form pairs. At position $p$ and frequency $\theta_r$:

$$
\begin{bmatrix}x'_{2r}\\x'_{2r+1}\end{bmatrix}
=\begin{bmatrix}\cos(p\theta_r)&-\sin(p\theta_r)\\
\sin(p\theta_r)&\cos(p\theta_r)\end{bmatrix}
\begin{bmatrix}x_{2r}\\x_{2r+1}\end{bmatrix},
\qquad \theta_r=10000^{-2r/d_h}.
$$

Rotations preserve the norm. Their transpose gives the backward pass.
Decode rotates new keys at their absolute positions once, stores the rotated
keys, and never rotates old cache entries again. This project uses adjacent
pairs; some model checkpoints use a different pairing convention.

## 8. Tiled online softmax

For one query tile, retain row maximum $m$, normalizer $\ell$, and an
unnormalized value accumulator $a$. Initialize
$m=-\infty,\ell=0,a=0$. A new key/value tile produces scores $S_b$:

$$
m'=\max(m,\operatorname{rowmax}(S_b)),\quad
\alpha=\exp(m-m'),\quad E_b=\exp(S_b-m').
$$

$$
\ell'=\alpha\ell+\operatorname{rowsum}(E_b),\qquad
a'=\alpha a+E_bV_b.
$$

After the last key tile, $O=a/\ell$. The rescaling keeps old and new terms
relative to the same maximum. The implementation handles all-masked tiles
with a finite exponential reference and returns zero if the final $\ell=0$.

Arithmetic work remains $O(BHL_qL_kd)$. Dense score and probability arrays
grow as $O(BHL_qL_k)$. Tiled working state grows as
$O(BH(B_qB_k+B_qd_v))$, plus the full output and the inputs. A caller-supplied
dense mask still occupies its own memory. The tiled path produces output
only; it has no backward implementation or full probability visualization.

This is an educational CPU implementation of tiled online attention, inspired
by the cited work. It does not reproduce FlashAttention's GPU kernel,
on-chip scheduling, or performance claims.

## Sources

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
