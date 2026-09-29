# Quiz — Ch. 5: Pretraining on unlabeled data

Self-test on the Ch. 5 material: loss and perplexity, the training loop, decoding
strategies, and checkpointing. Ten concise questions, one topic each — a short paragraph
is enough for any of them. Attempt them cold, write your answers, then ask me for
feedback.

---

**1.** Cross-entropy loss is described as the "negative average log probability". Walk
through what the model's output has to be turned into for that phrase to be literally
true, and say why the *negative* and the *log* are both there.

**2.** A loss of 10.79 corresponds to a perplexity of about 48,725. What does the
perplexity number mean in concrete terms, and why is it easier to interpret than the loss?

**3.** In `calc_loss_batch` the logits are reshaped with `logits.flatten(0, 1)` and the
targets with `target_batch.flatten()`. What shapes go in and come out, and why does
`cross_entropy` require this?

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
