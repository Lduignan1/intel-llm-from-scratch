# Chapter 3 test — Attention mechanisms (full chapter)

A 10-question test covering the whole of Ch. 3: from the intuition of attention through
self-attention, scaled dot-product, causal masking, dropout, and multi-head attention
(both the wrapper and the efficient single-matrix version).

Concept-focused. Write your answers under each question, then ask me for feedback.

---

**1.** At a high level, what problem does an attention mechanism solve that a plain
per-token embedding (or a fixed-window model) cannot? Frame it in terms of what
information a token's representation should capture.

> _Your answer:_ At a high level, an attention mechanism allows inputs to attend to all the other tokens in their context. Previous approaches only processed one token (or a small window) at a time and therefore struggled with handling long distance dependencies and more complex context dependent tasks. A token's representation should capture information about tokens that are likely to appear in its context.

**2.** Walk through the three steps that turn a sequence of input embeddings into context
vectors: scores → weights → output. For each step, say what operation is applied and what
the result represents.

> _Your answer:_ Step 1: the dot product is taken between the input embeddings and the weight matrices W_q, W_k and W_v to get the query, key and value vectors. The attention scores are computed by taking the dot product of the query and key vectors Step 2: We then divide/scale the result by the dimension size of the key vectors, d_k and then compute the attention weights by taking the softmax of the resulting vectors. Step 3: We take the dot product of the attention weights and the value vectors to get the final context vectors. 

**3.** Explain the distinct roles of the **query**, **key**, and **value** projections,
and give two reasons we learn three *separate* matrices rather than reusing one.

> _Your answer:_ The terms query, key and value are taken from concepts used in relational databases. The query is essential the "search term" or what we use to find what we are looking for. The key holds the info about the location about what we are searching for. Finally, the value holds the actual content of our search. We use three separate matrices instead of using one in order keep separate the weights which correspond to the distinct roles played the different projections. Having distinct query and key matrices means we can keep an asymetric relationship between the input token and the token we are comparing it with. Also, having a distinct value matrix means that we can decouple the relationship between two tokens and what the input token actually contributes itself.

**4.** Attention scores are divided by √d_k before the softmax. Explain *why the magnitude
of the scores grows with d_k* in the first place, and what specifically goes wrong in the
softmax (and in training) if you skip the scaling.

> _Your answer:_ The magnitude of the scores grows with d_k because they are computed by multiplying the values in Q and K matrices, and d_k is the dimension size of the vectors in K. Therefore the larger d_k, the larger the magnitude of the attention score vector. The larger the magnitude of the attention scores, the more likely the softmax compute will become saturated, similar to a one-hot vector which during training can lead to vanishing gradients.

**5.** What does "causal" mean in causal self-attention, why is it required for a
text-*generation* model, and at which point in the forward pass is the mask applied? Why
fill masked positions with `-inf` rather than `0`?

> _Your answer:_ "Causal" in self-attention means essential predicting the next token based on previous tokens. Another term used is "autoregressive". During training it involves applying a causal mask which blocks out future token values from the model. This is done usually between computing the attention scores and the attention weights for a given token. Masked positions are filled with `-inf` instead of `0` so that they do not influence the softmax computation. When computing softmax, we raise `e` to the power of the value in the input (attention score) vector. `e` raised to the power of 0 is equal to 1 and as a result any zero values will be turned into non zero values after application of the softmax.

**6.** In `CausalAttention` the mask is created with `register_buffer` rather than as an
`nn.Parameter`. Explain the practical difference between the two and why a buffer is the
right choice for the mask.

> _Your answer:_ The mask is created with `register_buffer` as this method does not require the model to store any numerical weights which would add to the total size of the model. This is in contrast with using `nn.Parameter`, whose additional gradients would add to the computational cost of training the model.

**7.** Dropout is applied to the attention weights. What is it doing conceptually, why is
it active during training but disabled at inference, and what does inverted-dropout
scaling accomplish?

> _Your answer:_

**8.** What is a "head," and why is multi-head attention useful compared to a single
larger attention computation? What role does the relationship `head_dim = d_out /
num_heads` play?

> _Your answer:_

**9.** Contrast `MultiHeadAttentionWrapper` with the efficient `MultiHeadAttention`. How
does each one produce the per-head results, and what does the single-matrix version do
(via `view`/`transpose`) to compute all heads at once? Mention the role of `out_proj`.

> _Your answer:_

**10.** The efficient `MultiHeadAttention` was *slower* than the wrapper in the notebook's
small-scale timing. Explain why an asymptotically more efficient implementation can lose
at small scale, and name the conditions under which it should win.

> _Your answer:_
