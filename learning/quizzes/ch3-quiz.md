# Quiz — Ch. 3: Attention mechanisms

Self-test on the Ch. 3 material. Attempt each cold, then ask me for feedback once
you've written your answers. Mix of recall, "why", and code-reasoning questions.

---

**1.** Define query, key, and value in one phrase each, in terms of the question each
"asks".

**2.** Why are attention scores divided by √d_k before the softmax? What goes wrong if
you omit it?

**3.** In the causal mask, future positions are set to `-torch.inf` rather than `0`.
Why `-inf`, and at what point in the computation does the masking happen?

**4.** `SelfAttention_v2` uses `nn.Linear` instead of the `nn.Parameter` matrices of
`v1`. Give one concrete reason this is an improvement, and explain why v1 and v2 produce
*different* outputs even on the same input.

**5.** In `MultiHeadAttention`, what does `head_dim = d_out // num_heads` imply, and why
is there an `assert d_out % num_heads == 0`?

**6.** What does the `out_proj` linear layer do in `MultiHeadAttention`, and why does
the naive `MultiHeadAttentionWrapper` not strictly need it?

**7.** (Code spot-check) These two lines come from different classes in your notebook:

```python
attn_scores.masked_fill_(mask, -torch.inf)   # CausalAttention
attn_scores.masked_fill(mask, -torch.inf)    # MultiHeadAttention
```

One of them has a bug. Which, and what's the consequence?

**8.** Why must you call `.contiguous()` before `.view()` after a `transpose()` in the
multi-head reshaping?

**9.** What makes this *self*-attention rather than cross-attention? Point to where in
the code you can tell.
