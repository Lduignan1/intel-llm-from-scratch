# Quiz — Ch. 4: GPT model components

Self-test on the Ch. 4 material. Attempt each cold, then ask me for feedback once
you've written your answers. Mix of recall, "why", and code-reasoning questions.

---

**1.** LayerNorm normalizes over `dim=-1`. Over what is it normalizing, and how does
that differ from Batch Norm?

**2.** What is the purpose of the learned `scale` and `shift` parameters in LayerNorm?
Couldn't we just always output mean-0/var-1?

**3.** Why does `LayerNorm` add `eps` (1e-5) inside the square root, and why use
`unbiased=False` for the variance?

**4.** Name two ways GELU differs from ReLU.

**5.** The FeedForward network expands to `4 * emb_dim` then back. Why expand at all,
and what work does FeedForward do that attention does *not*?

**6.** What is the purpose of residual (skip) connections in a transformer block?

**7.** A GPT model's final linear head outputs *logits*, not probabilities. Where/when
is softmax actually applied, and why keep it out of the forward pass?

**8.** What is the point of building `DummyGPTModel` before the real components?

**9.** Sketch the order of operations inside one Pre-LayerNorm transformer block (the
two sublayers, their norms, and where the residual adds happen).
