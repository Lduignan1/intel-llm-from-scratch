# Ch. 4 — GPT model components

> Maps to: `DummyGPTModel`/`DummyTransformerBlock`/`DummyLayerNorm` placeholders,
> then the real `LayerNorm`, `GELU`, `FeedForward` in the notebook.

## The "dummy first" pattern

`DummyGPTModel` wires up the **shape plumbing** (embeddings → N transformer blocks →
final norm → output head) with no-op stand-ins for the hard parts. This lets you verify
**tensor shapes flow end-to-end** before implementing real math. The output head maps
`emb_dim → vocab_size`, producing **logits** — one score per vocabulary token, for
every position.

## LayerNorm — stabilizing activations

Normalizes each token's feature vector to **mean 0, variance 1**, then applies a learned
rescale/reshift:

```python
mean = x.mean(dim=-1, keepdim=True)
var  = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * x_norm + self.shift
```

Key points:
- **`dim=-1`**: normalizes across the *feature* dimension, **per token**,
  independently of batch size. (This is Layer Norm, not Batch Norm — Batch Norm
  normalizes across the batch dim and behaves badly for variable-length sequences and
  small batches.)
- **`scale` and `shift`** (`nn.Parameter`, init to 1s and 0s): let the network *undo*
  normalization if that's optimal — it's not forced to keep mean 0/var 1.
- **`eps` (1e-5)**: prevents divide-by-zero when variance is tiny.
- **`unbiased=False`**: divides by `N`, not `N-1` (population variance). Matches GPT-2
  and keeps it consistent with the forward statistics.
- Why it matters: keeps activations in a stable range across deep stacks → faster, more
  reliable training.

## GELU — the activation function

Gaussian Error Linear Unit, tanh approximation (what GPT-2 uses):

```python
0.5 * x * (1 + tanh( √(2/π) * (x + 0.044715 · x³) ))
```

vs **ReLU** (`max(0, x)`):
- GELU is **smooth** (differentiable everywhere) — ReLU has a kink at 0.
- GELU allows **small negative outputs** instead of hard-zeroing — it weights inputs by
  how likely they are to be "kept" rather than gating on/off. Empirically better for
  transformers.

## FeedForward — per-token processing

A position-wise MLP applied identically to every token:

```python
nn.Sequential(
    nn.Linear(emb_dim, 4 * emb_dim),   # expand 4×
    GELU(),
    nn.Linear(4 * emb_dim, emb_dim),   # project back
)
```

- The **4× expansion** is the standard transformer ratio — a wider hidden space gives
  the model room to compute richer nonlinear features before compressing back.
- "Position-wise": no mixing *between* tokens here (that's attention's job). FeedForward
  refines each token's representation on its own.

## How they assemble (the transformer block — next in the notebook)

A GPT transformer block is, per the Pre-LayerNorm design:

```
x → LayerNorm → MultiHeadAttention → + x (residual)
  → LayerNorm → FeedForward        → + x (residual)
```

- **Residual (skip) connections** (`+ x`): add the input back to each sublayer's output.
  They give gradients a direct path through deep stacks, preventing vanishing gradients.
- **Pre-LN** (norm *before* each sublayer, as above) trains more stably than the
  original Post-LN design.

The full GPT = token+pos embeddings → `N` identical blocks → final LayerNorm →
linear head → logits → softmax (for sampling).

## Common pitfalls / things to verify

- LayerNorm normalizes `dim=-1` (features), **not** the batch or sequence dim.
- `scale`/`shift` must be `nn.Parameter`, not plain tensors, or they won't be learned.
- The FeedForward hidden layer is `4 * emb_dim`; getting this ratio wrong changes the
  parameter count substantially.
- Output is **logits**, not probabilities — softmax is applied at sampling/loss time,
  not inside the model's forward pass.

## Key terms

| Term | One-liner |
|------|-----------|
| Logits | Raw, pre-softmax scores (one per vocab token) |
| Layer Norm | Per-token feature normalization + learned scale/shift |
| GELU | Smooth activation; GPT-2's choice over ReLU |
| FeedForward / MLP | Per-token expand→activate→compress |
| Residual connection | `output + input`; gradient highway |
| Pre-LN | Normalize before the sublayer (stable training) |
