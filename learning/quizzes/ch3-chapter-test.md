# Chapter 3 test — Attention mechanisms (full chapter)

A 10-question test covering the whole of Ch. 3: from the intuition of attention through
self-attention, scaled dot-product, causal masking, dropout, and multi-head attention
(both the wrapper and the efficient single-matrix version).

Concept-focused. Write your answers under each question, then ask me for feedback.

---

**1.** At a high level, what problem does an attention mechanism solve that a plain
per-token embedding (or a fixed-window model) cannot? Frame it in terms of what
information a token's representation should capture.

> _Your answer:_ At a high level, an attention mechanism allows inputs to attend to all the other tokens in their context. Previous approaches only processed one token (or a small window) at a time and therefore struggled with handling long distance dependencies and more complex context dependent tasks. The attention mechanism is better because a token's representation should capture information about tokens that are likely to appear in its context and how other tokens influence it.

> **Feedback — ✅ Strong.** You've got it: direct access to all context tokens vs. the
> long-range struggle of window/sequential models. Two refinements: (1) the core payoff
> is a **context-dependent representation** — the same token ("bank") gets a *different*
> context vector depending on its neighbours; (2) the precise reason RNNs struggle isn't
> the window but the **fixed-size hidden-state bottleneck** (all past info squeezed
> through one vector); attention removes that by giving every token a direct path to
> every other.

**2.** Walk through the three steps that turn a sequence of input embeddings into context
vectors: scores → weights → output. For each step, say what operation is applied and what
the result represents.

> _Your answer:_ 
**Step 1**: the dot product is taken between the input embeddings and the weight matrices W_q, W_k and W_v to get the query, key and value vectors. The attention scores are computed by taking the dot product of the query and key vectors 
**Step 2***: We then divide the result by the dimension size of the key vectors, d_k, and then compute the attention weights by taking the softmax of the resulting vectors. 
**Step 3**: We take the dot product of the attention weights and the value vectors to get the final context vectors. 

> **Feedback — 🟡 One real error.** Steps 1 and 3 are correct. **Step 2 error: you divide
> by √d_k, not d_k.** In your code it's `attn_scores / keys.shape[-1]**0.5` — the `**0.5`
> is the square root. This ties directly to Q4 (score std grows like √d_k, so you cancel
> it with √d_k). Small wording point: Step 1 is a **matrix multiplication / linear
> projection** (`x @ W_q`), not a dot product — the dot products come *between* Q and K to
> form the scores.

**3.** Explain the distinct roles of the **query**, **key**, and **value** projections,
and give two reasons we learn three *separate* matrices rather than reusing one.

> _Your answer:_ The terms query, key and value are taken from concepts used in relational databases. The query is essential the "search term" or what we use to find what we are looking for. The key holds the info about the location of what we are searching for. Finally, the value holds the actual content of our search. We use three separate matrices instead of using one in order keep separate the weights which correspond to the distinct roles played the different projections. Having distinct query and key matrices means we can keep an asymetric relationship between the input token and the token we are comparing it with. Also, having a distinct value matrix means that we can decouple the relationship between two tokens and what the input token actually contributes itself.

> **Feedback — ✅✅ Excellent — this is the big win.** Exactly what you missed on the
> earlier quiz and now fully corrected: the database analogy, the **asymmetric Q≠K**
> relationship, *and* decoupling "how tokens match" from "what a token contributes" via a
> separate V. That concept underpins all of multi-head attention — great to see it locked
> in.

**4.** Attention scores are divided by √d_k before the softmax. Explain *why the magnitude
of the scores grows with d_k* in the first place, and what specifically goes wrong in the
softmax (and in training) if you skip the scaling.

> _Your answer:_ The magnitude of the scores grows with d_k because they are computed by multiplying the values in Q and K matrices, and d_k is the dimension size of the vectors in K. Therefore the larger d_k, the larger the magnitude of the attention score vector. The larger the magnitude of the attention scores, the more likely the softmax compute will become saturated, similar to a one-hot vector which during training can lead to vanishing gradients.

> **Feedback — ✅ Strong.** Correct end to end: magnitude grows with d_k → softmax
> saturates → peaky/near one-hot → vanishing gradients. To make it airtight: the reason
> magnitude grows is that each score is a **sum of d_k element-wise products**, so its
> *variance* scales with d_k and its **standard deviation with √d_k** — which is exactly
> why √d_k is the right divisor.

**5.** What does "causal" mean in causal self-attention, why is it required for a
text-*generation* model, and at which point in the forward pass is the mask applied? Why
fill masked positions with `-inf` rather than `0`?

> _Your answer:_ "Causal" in self-attention means essential predicting the next token based on previous tokens. Another term used is "autoregressive". During training it involves applying a causal mask which blocks out future token values from the model. This is done usually between computing the attention scores and the attention weights for a given token. Masked positions are filled with `-inf` instead of `0` so that they do not influence the softmax computation. When computing softmax, we raise `e` to the power of the value in the input (attention score) vector. `e` raised to the power of 0 is equal to 1 and as a result any zero values will be turned into non zero values after application of the softmax.

> **Feedback — ✅ Strong.** Meaning, autoregressive framing, mask-between-scores-and-
> softmax timing, and the `e^0 = 1` argument are all correct. One correction: causal
> masking isn't **training-only** — it's part of the architecture and applies at inference
> too (during generation the model only has past tokens anyway). So drop the "during
> training it involves" framing; the mask is always on in the forward pass.

**6.** In `CausalAttention` the mask is created with `register_buffer` rather than as an
`nn.Parameter`. Explain the practical difference between the two and why a buffer is the
right choice for the mask.

> _Your answer:_ The mask is created with `register_buffer` as this method does not require the model to store any numerical weights which would add to the total size of the model. This is in contrast with using `nn.Parameter`, whose additional gradients would add to the computational cost of training the model.

> **Feedback — ⚠️ Reframe — the framing is off.** It's not really about model *size* or
> gradient *cost* — a buffer still stores its full tensor. The real difference:
> - `nn.Parameter` → registered as **learnable**: appears in `model.parameters()`, gets
>   gradients, and the **optimizer updates it**.
> - `register_buffer` → part of module state (**moves with `.to(device)`**, saved in
>   `state_dict`) but **not** learnable — no gradient, never updated.
>
> Why a buffer fits the mask: the mask is a **fixed constant**, so it should never be
> trained (keep it out of `parameters()`) — but it **must still follow the model onto the
> GPU/XPU**, which a plain attribute tensor wouldn't. `register_buffer` = "device-tracked
> constant." That device-tracking benefit is the piece your answer is missing.

**7.** Dropout is applied to the attention weights. What is it doing conceptually, why is
it active during training but disabled at inference, and what does inverted-dropout
scaling accomplish?

> _Your answer:_ Conceptually, dropout is limiting the impact of a few weights on the computation of the context vectors, in order to avoid overfitting. We activate it during training because we do not want the model to become overreliant on a few attention weights as it is learning token representations. However, it is disabled at inference because we need all the attention weights when generating the highest probable token based on previous tokens. Inverted-dropout scaling scales up the unmasked values by a factor of 1/dropout_value in order to maintain the overall balance of the attention weights.

> **Feedback — ✅ Strong, one numeric fix.** Concept, train-vs-inference, and purpose all
> correct. **Fix the factor:** inverted dropout scales survivors by **1/(1 − p)**, where
> `p` is the *drop* probability — not `1/p`. (If p = 0.1, survivors are scaled by
> 1/0.9 ≈ 1.11, compensating for the ~10% zeroed so expected magnitudes match at
> inference.)

**8.** What is a "head," and why is multi-head attention useful compared to a single
larger attention computation? What role does the relationship `head_dim = d_out /
num_heads` play?

> _Your answer:_ A "head" is an attention layer. Using a multi-head architecture allows the model to attend to multiple different elements of an input sequence for a given token at once. A single larger attention computation would not permit this level of depth. Also existing GPU harware is optimized for a high number of parallel computations, making multihead attention the obvious choice. Each head has its own dimension size head_dim which when concatenated must equal the output dimension size d_out. Therefore, `head-dim = d_out / num_heads`.

> **Feedback — ✅ Strong.** "A head is an attention computation" (with its own Q/K/V into
> a subspace of size `head_dim`) and the concatenation logic are right, as is the GPU
> point. Refinement on *why* multi-head beats one big head: it's not "depth" — a single
> head produces **one** attention pattern (one averaging scheme), whereas multiple heads
> learn **several distinct patterns in parallel** (e.g. one tracks syntax, another
> long-range coreference). It's about diversity of relations, not depth.

**9.** Contrast `MultiHeadAttentionWrapper` with the efficient `MultiHeadAttention`. How
does each one produce the per-head results, and what does the single-matrix version do
(via `view`/`transpose`) to compute all heads at once? Mention the role of `out_proj`.

> _Your answer:_ `MultiHeadAttentionWrapper` initializes a list of attention heads which is then used to iteratively compute the context vectors one head at a time by calling the `CausalAttention` class. The `MultiHeadAttention` class improves upon this by leveraging matrix resizing operations and parallel dot product computations. In essence, the key, query and value matrices are split up by adding a new `num_heads` dimension with `view` and context vectors are computed from these higher dimension matrices via dimension swapping thanks to `transpose`. A final step is added to the pipeline with the application of the `out_proj` layer. This linear layer does not modify the shape of the output vectors but is added according to standard LLM training practice.  

> **Feedback — ✅ Mostly, but fix `out_proj`'s role.** The contrast (Python loop over
> `CausalAttention` vs. `view`/`transpose` batched heads) is well described. But `out_proj`
> is **not** just "standard practice" — it has a real job: after concatenation each
> position is just heads glued side by side with **no interaction between them**. The
> `out_proj` linear (`d_out → d_out`) **mixes information across heads**, letting the model
> learn how to combine what the different heads found. You're right it doesn't change the
> shape — but the *reason* it's there is cross-head mixing.

**10.** The efficient `MultiHeadAttention` was *slower* than the wrapper in the notebook's
small-scale timing. Explain why an asymptotically more efficient implementation can lose
at small scale, and name the conditions under which it should win.

> _Your answer:_ It a not fair to compare run time on a toy example of only two small input sequences. Here, the difference is only a few dozen milliseconds because the initialization of the the `MultiHeadAttention` takes slightly longer than the wrapper version. If we were to scale this up to larger batch sizes with hundreds or thousands of input tokens we would see a clear winner in efficiency with the `MultiHeadAttention` implementation.

> **Feedback — ✅ Good, deepen it.** Right instinct (toy scale, wins at scale). To make it
> precise: it's not mainly *initialization* — at small scale the **per-op overhead** (the
> `view`/`transpose`/`contiguous` reshapes, the extra `out_proj` matmul, Python dispatch)
> outweighs the tiny matmul it saves. The single-matrix win comes from **one large batched
> GEMM instead of N per-head kernel launches**, which only amortizes at large
> `num_heads`/sequence length **and especially on GPU/XPU**. Bonus: `%%time` is a single
> noisy run — `%timeit` would be a fairer measure.

---

## Scorecard

| Q | Topic | Verdict |
|---|-------|---------|
| 1 | Why attention | ✅ Strong |
| 2 | scores→weights→output | 🟡 √d_k, not d_k |
| 3 | Q/K/V roles | ✅✅ Nailed it (was the weak spot!) |
| 4 | √d_k scaling | ✅ Strong |
| 5 | Causal masking | ✅ Strong (mask isn't training-only) |
| 6 | buffer vs parameter | ⚠️ Reframe: learnable + device-tracking |
| 7 | Dropout | ✅ Strong (factor is 1/(1−p)) |
| 8 | Heads | ✅ Strong |
| 9 | Wrapper vs efficient | ✅ Fix `out_proj` = cross-head mixing |
| 10 | Perf puzzle | ✅ Good, deepen to per-op overhead |

**Overall: ~8.5/10 — a strong pass.** No major misconceptions left. The two worth
re-reading before Ch. 4 are **Q6** (buffer = non-learnable *but* device-tracked constant)
and **Q9** (`out_proj` mixes across heads). Everything else is small precision fixes
(√d_k, the 1/(1−p) factor).
