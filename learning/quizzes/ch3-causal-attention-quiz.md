# Quiz — Ch. 3 (up to the compact causal attention class)

Covers self-attention intuition, Q/K/V, scaled dot-product attention, causal masking,
and dropout — i.e. everything through `CausalAttention`, **before** multi-head attention.

Concept-focused. Write your answers under each question, then ask me for feedback.

---

**1.** In your own words, what is a *context vector*, and why is it a better
representation of a token than its raw input embedding?

> _Your answer:_ A context vector can be viewed as an enriched token embedding which encodes information from all other input tokens via the query, key and value projection matrices. They are better than raw input embeddings because they allow inputs to attend more or less to other tokens in context.  

> **Feedback — ✅ Strong.** You nailed the core idea: an enriched embedding that mixes in
> information from other tokens, weighted by relevance. One precision to lock in: a
> context vector is specifically a **weighted sum of the *value* vectors**, where the
> weights come from softmax over the scores. Your phrase "via the query, key and value
> projection matrices" slightly blurs roles — Q and K *produce the weights*; only **V** is
> what actually gets summed into the context vector. (Sets up Q2.)

**2.** Explain the distinct roles of the **query**, **key**, and **value** projections.
Why do we learn three separate matrices instead of attending directly on the raw
embeddings?

> _Your answer:_ These projection matrices allow us to compute the context vectors from the input embeddings. We use separate matrices in order to be able to train higher dimensions of parameters than just the input embeddings. The more parameters a model has, the more information it can store.

> **Feedback — ⚠️ Rework this one (most important concept in the chapter).**
> The "more parameters = more capacity" reason isn't the real one — and it's a bit off,
> since `d_out` can even be *smaller* than `d_in`. The actual reason is **functional
> differentiation** — each projection plays a distinct role:
> - **Query** = "what am I, this token, looking for in others?"
> - **Key** = "what do I, as a candidate, offer / how am I labeled?"
> - **Value** = "what content do I contribute if I get attended to?"
>
> The score for token *i* attending to *j* is `query_i · key_j` — compatibility between
> what *i* wants and what *j* advertises. Two reasons three separate matrices matter:
> 1. **Query ≠ Key** lets attention be **asymmetric** (A attending to B ≠ B attending to
>    A). One shared matrix would force a symmetric score matrix — too restrictive.
> 2. **Value separate from Key** decouples *what a token contributes* from *how it's
>    matched*. A token can be matched on one aspect but pass on different information.
>
> Rewrite around "three roles," not "more parameters."

**3.** Attention scores are scaled by `1/√d_k` before the softmax. What problem does this
scaling prevent, and what does it have to do with how the softmax behaves?

> _Your answer:_ Scaling the attention scores by `1/√d_k` helps prevent the softmax function from taking in extremely high values that might destabilize it.

> **Feedback — 🟡 Right idea, go one level deeper.** Add the *mechanism* and why it's
> √d_k specifically:
> - Dot products are sums over `d_k` dimensions, so their **variance grows with d_k** —
>   bigger head dimension → larger-magnitude scores purely from dimensionality.
> - Large scores make softmax **saturate** (output becomes peaky / near one-hot), and in
>   that regime **gradients vanish**, so learning stalls.
> - Dividing by √d_k rescales variance back to ~1, keeping softmax responsive regardless
>   of `d_k`. So it's not just "stability" in the abstract — it's about keeping gradients
>   alive as dimension scales.

**4.** What does "causal" mean in causal self-attention, and *why* is it required when
training a model to generate text? In the implementation, why are the masked positions
filled with `-inf` rather than simply zeroed out?

> _Your answer:_ Causal essential means "predict the next token based on previous tokens". It is essential to ensure a model learning how to generate text cannot see future tokens. Masked positions are filled with `-inf` as it is a simpler way to ensure the softmax function ignores them when normalizes the attention scores.

> **Feedback — ✅ Strong.** "Predict next from previous; can't see the future" is exactly
> right, and the leakage intuition (it'd be trivial/cheating during training) is the key
> point. Sharpen the `-inf` part by naming *when* in the pipeline it happens: masking is
> applied to the **scores, before softmax**. Since `exp(-inf) = 0`, those positions get
> **exactly zero weight while the surviving weights still renormalize to sum to 1**. If
> you set the *scores* to `0` instead, `exp(0) = 1` — future tokens would still get
> weight. (And zeroing the *weights* after softmax would break the sum-to-1 property.)

**5.** Dropout is applied to the **attention weights** in `CausalAttention`. What is
dropout doing conceptually, why is it applied during training but not inference, and what
would over-relying on a few attention weights look like without it?

> _Your answer:_ Conceptually dropout is removing/zeroing out a percentage of the attention weights to reduce reliance on just a few of them during training. It basically forces the model to work with less information in order to predict the correct token. This is only applied during training as the model is learning to optimize its weights in a constrained setting while during inference the model needs access to all its stored information in order to generate the "correct" tokens.

> **Feedback — ✅ Strong.** Conceptual purpose, the training-vs-inference split, and the
> constrained-learning framing are all correct. Two refinements:
> - The "without it" failure mode has a name: **co-adaptation / overfitting** — the model
>   leans on a few specific attention edges that fit the training data but don't
>   generalize.
> - At inference dropout isn't just "off" — standard **inverted dropout** scales the
>   surviving weights by `1/(1-p)` *during training* so expected magnitudes match, meaning
>   no rescaling is needed at inference.

---

## Scorecard

| Q | Topic | Verdict |
|---|-------|---------|
| 1 | Context vector | ✅ Strong (tighten: weighted sum of *values*) |
| 2 | Q/K/V roles | ⚠️ Rework — it's *roles*, not capacity |
| 3 | √d_k scaling | 🟡 Right, add the variance/gradient mechanism |
| 4 | Causal masking | ✅ Strong (name *when* the mask applies) |
| 5 | Dropout | ✅ Strong |

**Overall: ~4/5.** The only genuine misconception is Q2 — and since Q/K/V is *the*
foundation for multi-head attention (your next section), make sure that one really clicks
before moving on.
