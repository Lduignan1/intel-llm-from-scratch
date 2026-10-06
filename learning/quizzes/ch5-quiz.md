# Quiz — Ch. 5: Pretraining on unlabeled data

Self-test on the Ch. 5 material: loss and perplexity, the training loop, decoding
strategies, and checkpointing. Ten concise questions, one topic each — a short paragraph
is enough for any of them. Attempt them cold, write your answers, then ask me for
feedback.

---

**1.** Cross-entropy loss is described as the "negative average log probability". Walk
through what the model's output has to be turned into for that phrase to be literally
true, and say why the *negative* and the *log* are both there.
> For a given batch of inputs, the model outputs real values called logits stored in vectors of length vocab_size. These logits are turned into probabilities via the softmax function. We then take the probabilities of the target tokens and apply the logarithm to make these values more manageable for optimization. We then have a vector of log probabilities which we take the average of to get a singular value. As is common practice in machine learning, the average log probability is multiplied by -1 (to get a positive value) so that the goal becomes bringing the value down to 0 through training, instead of up from a negative one.

**2.** A loss of 10.79 corresponds to a perplexity of about 48,725. What does the
perplexity number mean in concrete terms, and why is it easier to interpret than the loss?

> Loss is an arbitrary value. The lower the better, but it can be hard to interpret it on its own. Perplexity is another metric used to evaluate the performance of language models. The perplexity score represents the number of token candidates that the model could generate at each next token prediction step. A perplexity of about 48,725 is obviously quite large and indicates that the model is very uncertain about what next token to generate. 

**3.** In `calc_loss_batch` the logits are reshaped with `logits.flatten(0, 1)` and the
targets with `target_batch.flatten()`. What shapes go in and come out, and why does
`cross_entropy` require this?

> The initial output of the GPTModel `logits`, is of shape [batch_size, seq_len, vocab_size]. For `cross_entropy` to work, we must combine the `logits` tensor over the batch size dimention with `logits.flatten(0, 1)` to get a shape of [batch_size x seq_len, vocab_size]. The `targets` tensor is transformed from size [batch_size, seq_len] to size [batch_size x seq_len]. The Pytorch function requires inputs of size [batch_size, num_classes] and targets of size [batch_size]. In our case, batch_size * seq_len is the effective batch size for Pytorch as it is a grouping of inputs which each are of length vocab_size, the number of possible classes (tokens) that can be predicted. 

**4.** You split "The Verdict" 90/10 into train and validation sets. What can the
validation loss tell you that the training loss alone cannot?

**5.** `evaluate_model` calls both `model.eval()` and `torch.no_grad()`. These do two
different things — what does each one switch off, and what would go wrong if you forgot
either?

**6.** Each batch in `train_model_simple` runs `optimizer.zero_grad()`, then
`loss.backward()`, then `optimizer.step()`. What breaks if you drop the `zero_grad()` call?

**7.** Training for 10 epochs on a 5,145-token corpus, the training loss keeps falling
while the validation loss flattens and then rises. What is happening, and why is this
dataset almost guaranteed to produce it?

**8.** `generate_text_simple` picks the next token with `argmax`. Replacing that with
`torch.multinomial` over the softmax probabilities changes the model's behaviour — what do
you gain, and what do you give up?

**9.** Temperature scaling divides the logits by a positive number before the softmax.
Describe the effect of a temperature above 1 versus below 1, and what the distribution
approaches at each extreme.

**10.** In `generate`, top-k sampling sets the non-selected logits to `-inf` *before* the
softmax, rather than zeroing their probabilities afterwards. Why is that the right place to
do it?
