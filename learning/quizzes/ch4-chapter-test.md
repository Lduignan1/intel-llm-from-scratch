# Chapter 4 test — Implementing a GPT model from scratch (full chapter)

A 5-question test covering the whole of Ch. 4: layer normalization, the GELU activation
and feed forward network, shortcut connections, the assembled transformer block, the full
`GPTModel` and its parameter count, and greedy text generation.

Because there are only five questions, each one bundles several related ideas — answer
every part.

Concept-focused. Write your answers under each question, then ask me for feedback.

---

**1.** `LayerNorm` normalizes `x` using `x.mean(dim=-1)` and `x.var(dim=-1)`. Explain
(a) *which* values get averaged together given an input of shape `(batch, tokens,
emb_dim)`, and why that grouping — rather than batch normalization's — is the right choice
for a language model; (b) what problem normalizing activations solves during training at
all; and (c) why `scale` and `shift` are `nn.Parameter`s when the whole point was to force
mean 0 and variance 1. Finally, what would break if `self.eps` were removed?

> _Your answer:_ (a) Given an input `x` of shape `(batch, tokens, emb_dim)` `LayerNorm` averages together the values of the embedding dimension. Batch normalization on the other hand normalizes inputs along the batch dimension. Batches sizes can vary greatly in LLM training depending on training infrastructure and compute resources. Since `LayerNorm` only normalizes along the feature dimension, the compute cost remains the same regardless of batch size. (b) Normalizing activations is done to stabilize model training by reducing the risk the vanishing or exploding gradients. Extreme values are compressed as embedding vectors have a variance of 1 and a mean of 0. (c) `scale` and `shift` are additional parameters that the model learns during training which are automatically adjusted if doing so increases performance. Removing `self.eps` risks breaking the computation of the normalized x input as it prevents division by 0.

> **Feedback — 🟡 (a) right answer, wrong reason; (c) is the real miss.**
> **(a)** Correct on *which* axis. Be more precise about the grouping: for shape
> `(batch, tokens, emb_dim)`, **each `(batch, token)` pair gets its own mean and variance**
> over its own 768 values — so `batch × tokens` independent statistics, no sharing at all.
> But the justification is off: it isn't about **compute cost** (a buffer of statistics is
> negligible either way). The real problem with BatchNorm here is *statistical dependence*:
> a token's normalized output would depend on **which other examples happen to share its
> batch**. That means (1) noisy estimates at small batch sizes, (2) running averages, so
> train and eval behave *differently*, and (3) at generation time batch is often **1**,
> where batch statistics are meaningless. LayerNorm draws its statistics from the single
> vector it is normalizing — identical behaviour at any batch size, and identical in
> training and inference.
> **(b)** ✅ Correct.
> **(c)** ⚠️ **This doesn't answer the question.** "Learned during training and adjusted if
> it improves performance" is true of *every* parameter in the model. The actual point:
> forcing mean 0 / variance 1 is a **constraint that destroys representational capacity** —
> maybe the next layer works better with a different scale or offset. `scale` and `shift`
> let the model **undo or modulate the normalization**, including learning the exact
> identity (`scale=1`, `shift=0` — which is precisely how they're initialized).
> Normalization stabilizes; scale/shift hand back the freedom it took away.
> **`eps`** ✅ Correct. Concretely: if a token's 768 values are all identical, `var = 0`,
> `sqrt(0) = 0`, and you divide by zero → `NaN` propagating through the whole model.

**2.** `FeedForward` expands `emb_dim` → `4 * emb_dim`, applies `GELU`, then projects back
down to `emb_dim`. Explain (a) what the expansion-and-contraction accomplishes that a
single `Linear(emb_dim, emb_dim)` could not; and (b) why GELU rather than ReLU — be
specific about what GELU does with *negative* inputs and why that property matters for
gradient-based learning. What is the significance of GELU being smooth everywhere?

> _Your answer:_ `FeedForward` expands `emb_dim` to compute richer non-linear features in a higher dimensional space that would not be possible with a single linear layer. GELU is used instead of ReLU as it is a smooth activation function while ReLU is simply max(0, x). This means that negative values are equal to zero and do not contribute to learning with ReLU. GELU on the other handle has nonzero gradients almost everywhere which means that even negative inputs values contribute to the training process, albeit to a lesser extent.

> **Feedback — ✅ Strong.** Both halves land.
> **(a)** Right idea, but sharpen *why* a single `Linear` can't match it. The reason isn't
> only width — it's that **two linear layers with nothing between them collapse
> algebraically into one**: `W₂(W₁x) = (W₂W₁)x`, still just one affine map. The **GELU in
> the middle** is what makes the sandwich strictly more expressive than any single linear
> layer. The 4× width then gives that nonlinearity enough dimensions to work with.
> **(b)** ✅ Correct on negatives and nonzero gradients. Name the failure mode you're
> describing — it's the **dying ReLU** problem: a unit pushed into `x < 0` receives exactly
> zero gradient, so it can never be updated back out again; it is dead *permanently*.
> GELU's small nonzero gradient lets such a unit recover.
> One conflation to untangle: you offered "nonzero gradients almost everywhere" as the
> significance of **smoothness**, but those are two different properties. Smoothness is
> about the **kink**: ReLU's derivative jumps discontinuously from 0 to 1 at `x = 0` (it
> isn't differentiable there), while GELU's derivative is continuous everywhere, giving a
> better-conditioned optimization landscape. Bonus fact worth knowing: GELU is
> **non-monotonic** — it dips slightly *below* zero around `x ≈ −0.75` before returning.

**3.** In the notebook, `print_gradients` on `ExampleDeepNeuralNetwork` gave gradient means
around `0.0002` for the first layer without shortcuts, versus `0.22` with them. Explain
(a) the mechanism that causes gradients to shrink as they propagate backwards through many
layers, and how adding `x + shortcut` changes what backprop sees; (b) why this matters more
for a 12-layer `GPTModel` than for a 2-layer network; and (c) in `TransformerBlock`, the
normalization is applied *before* the attention and feed forward sub-layers rather than
after. What is that arrangement called, and why is the input shape of a block necessarily
identical to its output shape?

> _Your answer:_ (a) Backpropagation is based on the chain rule in which partial derivatives of the weight parameters are multiplied all the way back through the network in order to compute gradients. Therefore, small values will get smaller and smaller the further back you go in the network and the opposite is true for large gradients. Adding `x + shortcut` after `MultiHeadAttention` and `FeedForward` modules of `TransformerBlock` helps prevent vanishing gradients by keeping them relatively large. (b) This matters more for 12-layer `GPTModel` than a 2-layer one as the vanishing gradients problem is exascerbated by deep models that have more gradients to compute. (c) The layer normalization implemented in `TransformerBlock` is known as Pre-LN, which leads to more stable training than applying it after the attention and FFN blocks. The input shape of a block must be identical to its output shape that multiple blocks can be stacked upon each other and so the output of one block becomes the input of the next. 

> **Feedback — ✅ Strong; sharpen the mechanism.**
> **(a)** Chain rule and repeated multiplication — correct. But "keeps them relatively
> large" describes the *effect*, not the *mechanism*. Write the derivative down: for
> `y = x + F(x)`, we get **`dy/dx = 1 + F'(x)`**. The identity path contributes exactly
> **1**, *additively*. So gradient reaches the earlier layer through a route that is never
> multiplied down, no matter how small `F'` is. That single `1` is the whole trick — and
> the reason your numbers went `0.0002 → 0.22`.
> **(b)** Reframe: it isn't "more gradients to compute" (a 12-layer net doesn't compute
> *more* gradients per weight). It's that the chain has **more factors multiplied
> together** — a product of N Jacobians, each with norm < 1, shrinks **exponentially in
> N**. 12 blocks × 2 sublayers ≈ 24 such factors versus 2. Exponential decay is why depth
> is the thing that breaks.
> **(c)** ✅ Pre-LN and the stacking argument are both right. There's a second reason for
> shape preservation you didn't mention, and it's the more fundamental one: `x + shortcut`
> is an **elementwise addition**, which is only defined if the sublayer's output has the
> same shape as its input. So identical in/out shape isn't merely convenient for stacking —
> it is **forced by the residual connection itself**.

**4.** Consider the assembled `GPTModel`. (a) Walk through what happens to `in_idx` from
token IDs to the final logits tensor, naming each component and saying why `tok_emb` and
`pos_emb` are *added* together. (b) Why does the output have shape
`(batch, tokens, vocab_size)` rather than `(batch, vocab_size)`? (c) `sum(p.numel() for p
in model.parameters())` gives ~163M, yet the model is called "GPT-2 124M" and the notebook
prints 124,412,160 only after subtracting `model.out_head.weight.numel()`. Explain what
accounts for the gap and what design decision in the original GPT-2 makes 124M the honest
figure. (d) The exercise found ~56.7M parameters in the feed forward modules versus ~28.3M
in the attention modules — roughly 2:1. Explain why that ratio falls out of the
architecture.

> _Your answer:_ (a) Tokenized input ids `in_idx` are first converted into trainable vector embeddings via the lookup table `tok_emb`. Positional embeddings `pos_emb` are then added to these tokens in order to encode information about the position of a token in a sequence. Some parameters are zeroed out via the embedding dropout module `drop_emb`. These embeddings are then passed through a series of `TransformerBlock`s resulting in context vectors. These outputs are passed through a final LayerNorm layer `final_norm` and then passed through a linear layer `out_head` to get vectors of length `vocab_size`. The values in these vectors are known as `logits`. (b) The output is of shape `(batch, tokens, vocab_size)` becuase each input token has a corresponding context vector, and there a corresponding set of output probabilities. (c) This discrepency is due to the fact that in the original GPT-2 model, the authors reused the token embedding layer for the output head to reduce the memory footprint of the model. (d) The 2:1 ratio of feed-forward modules to attention modules is a result of the larger sizes of the feed-forward networks which expand inputs to much larger sizes than the attention modules.

> **Feedback — ✅ Good, with one slip and two gaps.**
> **(a)** The pipeline is correct end to end — `tok_emb` → `+ pos_emb` → `drop_emb` →
> 12 × `TransformerBlock` → `final_norm` → `out_head` → logits. Two notes:
> - **Slip:** "Some **parameters** are zeroed out via the embedding dropout." Dropout zeroes
>   **activations** — the values flowing through the network — never the parameters. The
>   weights are untouched; only that forward pass's signal is masked.
> - **Gap:** the question asked why they are **added** rather than concatenated. Addition
>   keeps the width at `emb_dim`; concatenating would give 1536 and **double the cost of
>   every downstream layer**. The model can learn to keep the two signals separable inside
>   the shared 768 dimensions, so addition is both sufficient and cheaper.
>
> **(b)** ✅ Right. One precision point: they're **logits**, not probabilities, until a
> softmax is applied — you used the correct term yourself in (a).
> **(c)** ✅✅ Correct, and this is the one people usually miss. The name for it is **weight
> tying**. Verified: `out_head` = 768 × 50,257 = **38,597,376**, and
> 163,009,536 − 38,597,376 = **124,412,160** — exactly the notebook's figure.
> **(d)** Right instinct, but the answer is *exact* rather than vague, so do the counting:
> - `FeedForward`: **2** matrices of `d × 4d` → **8d²**
> - `MultiHeadAttention`: **4** matrices (`W_q`, `W_k`, `W_v`, `out_proj`) of `d × d` → **4d²**
> - 8d² / 4d² = **exactly 2** — reproducing 56,669,184 vs 28,320,768.
>
> The subtlety worth stating: attention has **more** matrices (4 vs 2), but each is
> **smaller** (`d×d` vs `d×4d`) — and the 4× width wins.

**5.** Consider `generate_text_simple`. (a) Why must the function loop and re-run the model
for each new token, rather than producing `max_new_tokens` outputs in a single forward
pass? (b) The model returns a distribution at *every* position, but the code keeps only
`logits[:, -1, :]` — explain why the other positions are discarded here, and in what
situation (later in the book) those discarded positions become the useful part. (c) The
code computes `torch.softmax(...)` and then immediately takes `torch.argmax` of the result.
Softmax is strictly monotonic — so is this step mathematically necessary? Justify your
answer, and say why it is nevertheless reasonable to leave it in. (d) Selecting the
highest-probability token every time is called *greedy decoding*. Name two concrete
weaknesses of always taking the argmax when generating text.

> _Your answer:_ (a) The model is only capable of predicting one token at a time. For each new token, the model needs to take as input all the tokens including the most recent generated token in its context window to make the next prediction. (b) The code keeps only the last position because it is from this distribution that the next token is created. The other positions are not needed for inference, but they are useful for training. (c) It is reasonable to keep softmax because it returns a probability distribution which leads to more stable training by giving clean gradients. (d) Greedy decoding can lead to (1) a lack of creativity in outputs and (2) repetition loops where the model gets stuck repeating the same word or phrase.

> **Feedback — 🟡 (c) is wrong; the rest is solid.**
> **(a)** ✅ Correct — the autoregressive feedback loop, stated cleanly.
> **(b)** ✅ Correct. Worth deepening: in **training** all positions are used *at once* —
> one forward pass over n tokens yields **n prediction targets** (row i predicts token i+1),
> which is exactly what makes training efficient. That's teacher forcing, and it's why the
> model is built to emit a distribution at every position even though generation reads one.
> **(c)** ❌ **This misses the question, and the justification doesn't hold.** You were
> asked whether the step is *mathematically necessary*. The answer is **no**. Softmax is
> strictly monotonic, so it preserves the ordering of the scores: `argmax(softmax(z))` and
> `argmax(z)` return the same index, always — I verified it over 200 random trials on
> `(4, 50257)` tensors. The line is **pure redundant computation** with no effect on the
> output. Your reasoning — "more stable training by giving clean gradients" — doesn't apply
> here: **there is no training in this function at all.** It runs entirely under
> `torch.no_grad()`; no gradients exist to be stabilized.
> Why it's still reasonable to leave in: (1) it makes the intent legible — "pick the most
> probable token" — and (2) it's the **hook for Chapter 5**, where temperature scaling and
> top-k sampling need genuine probabilities and `argmax` gets swapped for
> `torch.multinomial`. Keeping it means that upgrade is a one-line change.
> **(d)** ✅ Both correct and concrete. A third worth adding: **determinism** — the same
> prompt always produces byte-identical output, so you cannot sample several candidates and
> pick the best.

---

## Scorecard

| Q | Topic | Verdict |
|---|-------|---------|
| 1 | LayerNorm | 🟡 (c) missed — scale/shift restore lost capacity |
| 2 | GELU + FeedForward | ✅ Strong (linear layers collapse without the nonlinearity) |
| 3 | Shortcuts + TransformerBlock | ✅ Strong — sharpen to `1 + F'(x)` |
| 4 | GPTModel + parameter counts | ✅ Good — weight tying nailed; dropout zeroes *activations* |
| 5 | Text generation | 🟡 (c) softmax **is** mathematically redundant |

**Overall: ~7.5/10 — a solid pass.** No structural misunderstandings: the pipeline, the
residual mechanism, weight tying, and greedy decoding's weaknesses are all genuinely
understood. The two worth re-reading before Ch. 5:

1. **Q1(c)** — normalization *removes* representational freedom; `scale`/`shift` give it
   back. "It's learned" isn't an explanation on its own.
2. **Q5(c)** — check the actual question being asked. "Is this necessary?" wants a yes/no
   plus a reason, and here the honest answer is *no, it's redundant, and it's kept for
   readability and forward-compatibility*.

Recurring habit to watch: several answers gave the **effect** where the question asked for
the **mechanism** (Q3a "keeps gradients large" vs `1 + F'(x)`; Q4d "feed forward is bigger"
vs 8d² vs 4d²). When a number or a formula is available, reach for it — it's almost always
what the question is fishing for.
