# Quiz — Ch. 3 (up to the compact causal attention class)

Covers self-attention intuition, Q/K/V, scaled dot-product attention, causal masking,
and dropout — i.e. everything through `CausalAttention`, **before** multi-head attention.

Concept-focused. Write your answers under each question, then ask me for feedback.

---

**1.** In your own words, what is a *context vector*, and why is it a better
representation of a token than its raw input embedding?

> _Your answer:_ A context vector can be viewed as an enriched token embedding which encodes information from all other input tokens via the query, key and value projection matrices. They are better than raw input embeddings because they allow inputs to attend more or less to other tokens in context.  

**2.** Explain the distinct roles of the **query**, **key**, and **value** projections.
Why do we learn three separate matrices instead of attending directly on the raw
embeddings?

> _Your answer:_ These projection matrices allow us to compute the context vectors from the input embeddings. We use separate matrices in order to be able to train higher dimensions of parameters than just the input embeddings. The more parameters a model has, the more information it can store.

**3.** Attention scores are scaled by `1/√d_k` before the softmax. What problem does this
scaling prevent, and what does it have to do with how the softmax behaves?

> _Your answer:_ Scaling the attention scores by `1/√d_k` helps prevent the softmax function from taking in extremely high values that might destabilize it.

**4.** What does "causal" mean in causal self-attention, and *why* is it required when
training a model to generate text? In the implementation, why are the masked positions
filled with `-inf` rather than simply zeroed out?

> _Your answer:_ Causal essential means "predict the next token based on previous tokens". It is essential to ensure a model learning how to generate text cannot see future tokens. Masked positions are filled with `-inf` as it is a simpler way to ensure the softmax function ignores them when normalizes the attention scores.

**5.** Dropout is applied to the **attention weights** in `CausalAttention`. What is
dropout doing conceptually, why is it applied during training but not inference, and what
would over-relying on a few attention weights look like without it?

> _Your answer: Conceptually dropout is removing/zeroing out a percentage of the attention weights to reduce reliance on just a few of them during training. It basically forces the model to work with less information in order to predict the correct token. This is only applied during training as the model is learning to optimize its weights in a constrained setting while during inference the model needs access to all its stored information in order to generate the "correct" tokens.
