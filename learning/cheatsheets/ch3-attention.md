# Ch. 3 — Attention mechanisms

> Maps to: `SelfAttention_v1` → `SelfAttention_v2` → `CausalAttention` →
> `MultiHeadAttentionWrapper` → `MultiHeadAttention` in the notebook.
> Each class is a refinement of the last — that progression *is* the lesson.

## The core idea

Each token builds a new representation (a **context vector**) by **weighted-averaging
information from other tokens**. The weights say "how much should I attend to you?"
and are computed from the tokens themselves. That's it — everything else is mechanics
and efficiency.

## Query / Key / Value (the central abstraction)

For every token's embedding `x`, project it three ways with learned matrices:

| Role | Meaning | Code |
|------|---------|------|
| **Query** (Q) | "What am I looking for?" | `queries = W_query(x)` |
| **Key** (K) | "What do I offer / how am I labeled?" | `keys = W_key(x)` |
| **Value** (V) | "What content do I pass on if attended to?" | `values = W_value(x)` |

The three steps:

```
scores  = Q @ Kᵀ                      # similarity of each query to each key
weights = softmax(scores / √d_k)      # normalize into a probability distribution
context = weights @ V                 # weighted sum of values
```

### Why `/ √d_k` (scaled dot-product attention)?

Dot products grow with dimension `d_k`. Large magnitudes push softmax into saturated
regions where gradients vanish. Dividing by `√d_k` keeps the variance ~1, so softmax
stays in a well-behaved range. In your code: `attn_scores / keys.shape[-1]**0.5`.

## The progression in your notebook

### `SelfAttention_v1` — raw `nn.Parameter`
- Q/K/V are `nn.Parameter(torch.rand(d_in, d_out))`, applied as `x @ W`.
- Teaches the math with nothing hidden.

### `SelfAttention_v2` — `nn.Linear`
- Swaps the raw matrices for `nn.Linear(d_in, d_out, bias=...)`.
- Why it's better: `nn.Linear` has a **smarter weight initialization** and integrates
  cleanly with the optimizer. (Note: the learned matrices differ from v1, so outputs
  won't match unless you copy weights across — a classic exercise.)

### `CausalAttention` — masking + dropout + batching
Two additions that make it usable for a real GPT:

1. **Causal mask** — a token must not see the future (it's predicting the future).
   Upper-triangular positions are set to `-∞` *before* softmax, so they become 0 after:
   ```python
   self.register_buffer('mask',
       torch.triu(torch.ones(context_length, context_length), diagonal=1))
   ...
   attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
   ```
   - `-∞` not `0`, because masking happens **before** softmax; `exp(-∞) = 0`.
   - `register_buffer` → the mask moves with `.to(device)` but is **not** a learned
     parameter (no gradient).
2. **Dropout on the attention weights** — randomly zero some weights during training for
   regularization.
3. Handles a **batch** dim: `Q @ Kᵀ` becomes `queries @ keys.transpose(1, 2)`.

### `MultiHeadAttentionWrapper` — heads in parallel, naively
- Runs `num_heads` independent `CausalAttention` modules and **concatenates** their
  outputs: `torch.cat([head(x) for head in self.heads], dim=-1)`.
- Each head can specialize (one tracks syntax, another long-range coreference, etc.).
- Correct but **slow** — a Python loop and separate matrices per head.

### `MultiHeadAttention` — the efficient single-matrix version
Same result, one set of big matrices, no loop:

1. Project once to full `d_out`, then **split into heads by reshaping**:
   `[b, tokens, d_out] → view → [b, tokens, num_heads, head_dim] → transpose → [b, num_heads, tokens, head_dim]`.
   where `head_dim = d_out // num_heads` (hence the `assert d_out % num_heads == 0`).
2. Batched attention across the head dimension at once.
3. Recombine heads (`transpose` back, `.contiguous().view(b, tokens, d_out)`).
4. **Output projection** `out_proj` — a final `nn.Linear(d_out, d_out)` that mixes
   information across heads. (The wrapper version lacked this.)

## Common pitfalls / things to verify

- **`masked_fill` vs `masked_fill_`**: the trailing underscore = in-place. A
  non-in-place `masked_fill` returns a new tensor; if you don't assign it, the mask is
  silently discarded and the model can attend to the future. Check your
  `MultiHeadAttention.forward` carefully — `CausalAttention` uses the in-place form.
- `.contiguous()` before `.view()` after a `transpose`: `transpose` returns a
  non-contiguous view, and `.view()` requires contiguous memory.
- Mask is sliced to `[:num_tokens, :num_tokens]` so shorter sequences than
  `context_length` still work.
- Q/K/V projections are computed from the **same** input `x` — that's what makes it
  *self*-attention (vs. cross-attention, where Q comes from one source and K/V another).

## Key terms

| Term | One-liner |
|------|-----------|
| Self-attention | Q, K, V all come from the same sequence |
| Causal / masked attention | Future positions masked out (autoregressive) |
| Attention weights | softmax-normalized scores; rows sum to 1 |
| Head | One independent Q/K/V attention computation |
| `head_dim` | `d_out / num_heads` |
| Output projection | Final linear that mixes the concatenated heads |
| `register_buffer` | Non-trainable tensor that still tracks device |
