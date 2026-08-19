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

**2.** `FeedForward` expands `emb_dim` → `4 * emb_dim`, applies `GELU`, then projects back
down to `emb_dim`. Explain (a) what the expansion-and-contraction accomplishes that a
single `Linear(emb_dim, emb_dim)` could not; and (b) why GELU rather than ReLU — be
specific about what GELU does with *negative* inputs and why that property matters for
gradient-based learning. What is the significance of GELU being smooth everywhere?

> _Your answer:_ `FeedForward` expands `emb_dim` to compute richer non-linear features in a higher dimensional space that would not be possible with a single linear layer. GELU is used instead of ReLU as it is a smooth activation function while ReLU is simply max(0, x). This means that negative values are equal to zero and do not contribute to learning with ReLU. GELU on the other handle has nonzero gradients almost everywhere which means that even negative inputs values contribute to the training process, albeit to a lesser extent.

**3.** In the notebook, `print_gradients` on `ExampleDeepNeuralNetwork` gave gradient means
around `0.0002` for the first layer without shortcuts, versus `0.22` with them. Explain
(a) the mechanism that causes gradients to shrink as they propagate backwards through many
layers, and how adding `x + shortcut` changes what backprop sees; (b) why this matters more
for a 12-layer `GPTModel` than for a 2-layer network; and (c) in `TransformerBlock`, the
normalization is applied *before* the attention and feed forward sub-layers rather than
after. What is that arrangement called, and why is the input shape of a block necessarily
identical to its output shape?

> _Your answer:_ (a) Backpropagation is based on the chain rule in which partial derivatives of the weight parameters are multiplied all the way back through the network in order to compute gradients. Therefore, small values will get smaller and smaller the further back you go in the network and the opposite is true for large gradients. Adding `x + shortcut` after `MultiHeadAttention` and `FeedForward` modules of `TransformerBlock` helps prevent vanishing gradients by keeping them relatively large. (b) This matters more for 12-layer `GPTModel` than a 2-layer one as the vanishing gradients problem is exascerbated by deep models that have more gradients to compute. (c) The layer normalization implemented in `TransformerBlock` is known as Pre-LN, which leads to more stable training than applying it after the attention and FFN blocks. The input shape of a block must be identical to its output shape that multiple blocks can be stacked upon each other and so the output of one block becomes the input of the next. 

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

> _Your answer:_
