# Quiz — Ch. 5: Pretraining on unlabeled data

Self-test on the Ch. 5 material: loss and perplexity, the training loop, decoding
strategies, and checkpointing. Ten concise questions, one topic each — a short paragraph
is enough for any of them. Attempt them cold, write your answers, then ask me for
feedback.

**Marked:** 8 solid, 2 needing work (Q5, Q8). Summary at the bottom.

---

**1.** Cross-entropy loss is described as the "negative average log probability". Walk
through what the model's output has to be turned into for that phrase to be literally
true, and say why the *negative* and the *log* are both there.
> For a given batch of inputs, the model outputs real values called logits stored in vectors of length vocab_size. These logits are turned into probabilities via the softmax function. We then take the probabilities of the target tokens and apply the logarithm to make these values more manageable for optimization. We then have a vector of log probabilities which we take the average of to get a singular value. As is common practice in machine learning, the average log probability is multiplied by -1 (to get a positive value) so that the goal becomes bringing the value down to 0 through training, instead of up from a negative one.

> **Feedback — ✅ The pipeline is exactly right.** logits → softmax → gather the target
> token's probability → log → mean → negate. That's the whole chain, in order.
> The one soft spot is **why the log**. "More manageable for optimization" is vague — there
> are three concrete reasons:
> 1. **Products become sums.** The likelihood of a sequence is the *product* of per-token
>    probabilities. For 1,024 tokens each around 1e-5, that product **underflows to 0.0** in
>    float32. `log` turns it into a sum of ~1,024 manageable terms, which is the only
>    numerically viable form.
> 2. **The gradients are better behaved.** Through a softmax, the gradient of the log
>    probability is the clean `(predicted − one-hot target)`. Probability-space loss
>    saturates: a token predicted at 1e-8 gets an almost flat gradient, exactly where you
>    most need a strong signal.
> 3. **It's monotonic**, so maximizing log-likelihood is the same problem as maximizing
>    likelihood — you lose nothing by transforming.
> On the negation: ✅ right, and the framing is the standard one — optimizers *minimize*, so
> you flip maximize-log-likelihood into minimize-negative-log-likelihood. Small precision
> point: the floor is 0 but it is **unreachable in practice**, since it requires assigning
> probability exactly 1.0 to every target token.
> Worth knowing: `cross_entropy` doesn't actually run softmax then log as two steps — it
> uses `log_softmax` internally (the log-sum-exp trick), which is why you must hand it **raw
> logits**, never softmax output. Passing probabilities is a common silent bug.

**2.** A loss of 10.79 corresponds to a perplexity of about 48,725. What does the
perplexity number mean in concrete terms, and why is it easier to interpret than the loss?

> Loss is an arbitrary value. The lower the better, but it can be hard to interpret it on its own. Perplexity is another metric used to evaluate the performance of language models. The perplexity score represents the number of token candidates that the model could generate at each next token prediction step. A perplexity of about 48,725 is obviously quite large and indicates that the model is very uncertain about what next token to generate. 

> **Feedback — ✅ Correct, and the interpretation lands.**
> Tighten the definition slightly: perplexity is the size of the set of tokens the model is
> effectively choosing between **uniformly at random**. It's not "how many candidates exist"
> — it's an *effective* count. A model that puts 0.5 on one token and spreads the rest
> thinly has a far lower perplexity than one spreading 0.5 over 100 tokens, even though both
> have thousands of nonzero candidates.
> The number you quoted is the punchline worth remembering: vocab size is **50,257**, and
> perplexity is **48,725** — the untrained model is picking almost uniformly from the entire
> vocabulary. Perplexity ≈ vocab size is the signature of a model that has learned nothing.
> One phrase to retire: loss isn't "arbitrary" — it has units (nats per token), and `exp` of
> it is exactly the perplexity. The reason perplexity reads more easily is that it lives in
> **token-count units** you can picture, and it's **multiplicative**: a loss drop from 10.79
> to 9.79 (one nat) is a 2.7× reduction in perplexity. Equal-sized loss drops are equal
> *ratios* of perplexity, which is why loss curves look deceptively flat late in training.

**3.** In `calc_loss_batch` the logits are reshaped with `logits.flatten(0, 1)` and the
targets with `target_batch.flatten()`. What shapes go in and come out, and why does
`cross_entropy` require this?

> The initial output of the GPTModel `logits`, is of shape [batch_size, seq_len, vocab_size]. For `cross_entropy` to work, we must combine the `logits` tensor over the batch size dimention with `logits.flatten(0, 1)` to get a shape of [batch_size x seq_len, vocab_size]. The `targets` tensor is transformed from size [batch_size, seq_len] to size [batch_size x seq_len]. The Pytorch function requires inputs of size [batch_size, num_classes] and targets of size [batch_size]. In our case, batch_size * seq_len is the effective batch size for Pytorch as it is a grouping of inputs which each are of length vocab_size, the number of possible classes (tokens) that can be predicted. 

> **Feedback — ✅ Fully correct, including the conceptual point.**
> Shapes match your notebook exactly: `[2, 3, 50257]` → `[6, 50257]`, and `[2, 3]` → `[6]`.
> The key insight — that `batch × seq_len` becomes the effective batch of independent
> classification problems — is the right way to think about it. A GPT training step is
> literally `batch × seq_len` separate 50,257-way classifications, averaged.
> Two refinements:
> - `flatten(0, 1)` **merges dims 0 and 1** (it's a range of dims to collapse, not "combine
>   over the batch dimension"). `flatten()` with no args collapses everything, which is why
>   the targets call needs no arguments.
> - PyTorch names those arguments `(N, C)` and `(N,)`. `N` isn't required to be your
>   DataLoader batch size — any leading dimension works as long as logits and targets agree
>   on it. Worth knowing: `cross_entropy` also accepts `(N, C, d1)` directly, so
>   `logits.transpose(1, 2)` with the targets unflattened would compute the same number. The
>   flatten form is just the clearer one.

**4.** You split "The Verdict" 90/10 into train and validation sets. What can the
validation loss tell you that the training loss alone cannot?

> The validation set tells you how the model performs on unseen, in-domain data. The loss on the train data is expected to naturally decrease over the specified number of epochs as the model is being tested on the examples it has been trained on, effectively memorizing them. The loss on the validation set generally flattens out before and remains above that of the train set. The validation score is a better indication of how the model will perform in practice, on in domain data. 

> **Feedback — ✅ Right, and you named the crucial asymmetry.**
> The core point is exactly this: training loss **cannot distinguish learning from
> memorization**, because both drive it down. Only held-out data separates them.
> Two things to add that make it operationally useful:
> - **The gap itself is the diagnostic, not either curve alone.** Training loss falling with
>   validation loss tracking it = genuine learning. Training loss falling while validation
>   loss rises = memorization. You read the *divergence*.
> - It gives you a **stopping criterion**. The epoch where validation loss bottoms out is
>   roughly where to stop (early stopping); past it you're training a model that is getting
>   worse at its actual job while the training curve keeps looking encouraging.
> One caveat worth internalizing now, since it bites later: once you start **choosing**
> things based on validation loss (epochs, learning rate, architecture), information from
> that set leaks into your model and the validation loss becomes mildly optimistic. That's
> precisely why serious setups keep a third **test set**, touched once at the end.

**5.** `evaluate_model` calls both `model.eval()` and `torch.no_grad()`. These do two
different things — what does each one switch off, and what would go wrong if you forgot
either?

> Setting `model.eval()` turns off dropout which is an option that zeroes out a specified percentage of neurons in the network. This is useful during training when we want to avoid cases in which the model overrelies on a few neurons. It is however not useful during model inference and could actually hurt performance. The other setting, `torch.no_grad()` turns off tensor gradient accumulation which is needed for backpropagation during training, but would unncessarily add extra compute overhead (therefore a slowdown) during inference.

> **Feedback — 🟡 Both halves identify the right mechanism, but the consequences are
> understated — and the second question ("what would go wrong") is only half answered.**
> **`model.eval()`:** correct that it disables dropout. Two additions:
> - Dropout doesn't only zero units, it **rescales the survivors by `1/(1-p)`** so the
>   expected magnitude of the activations is preserved. Without that rescaling, turning
>   dropout off at inference would systematically inflate every activation.
> - It is a **flag**, not an operation: `self.training = False` on every submodule. It
>   changes dropout and (in other architectures) BatchNorm's switch to running statistics.
>   It does **not** touch autograd at all — that's the whole reason you need both calls.
> - What goes wrong if you forget it: your loss becomes **noisy and random** — dropout is
>   resampled every forward pass, so the same batch gives a different (and biased-upward)
>   loss each time. Your train-vs-validation comparison stops being apples-to-apples, and
>   generated text degrades.
> **`torch.no_grad()`:** right that it stops the autograd graph being built. But "a
> slowdown" undersells it badly, and **memory is the real cost** — as this project found out
> the hard way. Without `no_grad`, every intermediate activation in all 12 transformer
> blocks is **retained** for a backward pass that never comes. On a 1024-token context
> that's gigabytes, and on your 15 GB machine it is the difference between evaluating fine
> and OOM-ing. Note the numbers it produces are still *correct* — the failure mode is
> resource exhaustion, not wrong results.
> **The clean mental split, worth memorizing:** `eval()` changes *what the model computes*;
> `no_grad()` changes *what PyTorch remembers*. Neither implies the other.

**6.** Each batch in `train_model_simple` runs `optimizer.zero_grad()`, then
`loss.backward()`, then `optimizer.step()`. What breaks if you drop the `zero_grad()` call?

> If you drop the `zero_grad()` call, then Pytorch will not delete the accumulated gradients from the previous optimizer step. The next weight update will then be perturbed by the previously saved gradients and training will collapse.

> **Feedback — ✅ Correct mechanism.** `backward()` **adds into** `param.grad` rather than
> overwriting it, so without zeroing you carry the old gradients forward.
> Make the consequence precise, because it's more specific than "perturbed": after `k`
> batches with no zeroing, `param.grad` holds the **running sum of all `k` gradients so
> far**. Two distinct problems follow:
> - The **effective learning rate grows without bound** — step magnitude scales with `k`.
> - The update is **stale**: it is dominated by gradients computed at parameter values the
>   model has long since left behind.
> In practice you see the loss spike and then go `NaN` within a few dozen steps, so
> "collapse" is fair — just know *why* it collapses.
> The part worth adding: this accumulation is **deliberate, not a footgun**. It's what makes
> **gradient accumulation** possible — run `backward()` over several micro-batches, then one
> `step()`, and you get the gradient of a large batch on a GPU that can't hold one. Directly
> relevant to you: it's how you'd train with an effective batch size of 16 on a machine that
> only fits 2. The order also matters for a subtler reason: `step()` must come **after**
> `backward()` because it reads `param.grad`, and `zero_grad()` must come **before**
> `backward()` and never between `backward()` and `step()` — that would hand the optimizer
> all-zero gradients and silently train nothing.

**7.** Training for 10 epochs on a 5,145-token corpus, the training loss keeps falling
while the validation loss flattens and then rises. What is happening, and why is this
dataset almost guaranteed to produce it?

> The reason why the loss keeps falling on the train corpus but flattens and starts to go up on the validation is because the model is overfitting (memorizing the training data). A 5,145 token corpus is extremely small for a pretraining corpus as models are generally trained on billions or even trillions of tokens at this stage in order to learn sufficient general capabilities.

> **Feedback — ✅ Correct diagnosis and the right comparison.**
> Make the "almost guaranteed" part quantitative, because the arithmetic is striking: your
> model has **163M parameters** and the corpus has **5,145 tokens** — roughly **31,000
> parameters per training token**. There is vastly more capacity to store the text than
> there is text, so the lowest-loss solution available to the optimizer *is* memorization.
> Overfitting here isn't a risk to be managed, it's the expected outcome.
> Two things to add:
> - With `max_length=256` and `stride=256`, the 4,612 training tokens give **18 chunks, so
>   9 batches of 2** — the whole epoch is 9 optimizer steps.
>   Ten epochs means the model sees every token ten times — and you can watch the
>   memorization directly in the generated samples, which start reproducing Wharton's exact
>   sentences.
> - It's worth being explicit that the book **intends** this. The purpose of this run is to
>   verify the training loop works end to end on a laptop, not to produce a good model. The
>   real fix isn't regularization, it's **more data** — which is exactly why the chapter then
>   pivots to loading OpenAI's pretrained weights instead of training longer.

**8.** `generate_text_simple` picks the next token with `argmax`. Replacing that with
`torch.multinomial` over the softmax probabilities changes the model's behaviour — what do
you gain, and what do you give up?

> By replacing `argmax` with `torch.multinomial` in `generate_text_simple`, we switch from deterministic to probabilistic token sampling. Instead of always generating the most likely token from the softmax distribution, we now select tokens proportional to their probability score. For example, token "a" may have a score of 0.9 and "the" might have one of 0.05. With `argmax`, "the" would always be chosen as the next token whereas with `multinomial` it would instead be chosen 90% of the time, with the possibility of "the" being chosen 5% of the time. In the latter case, we gain an increase in unique, creative outputs while giving up some consistency in responses.  

> **Feedback — 🟡 The concept is right; the worked example is garbled, and the trade-off
> needs sharpening.**
> **The slip:** you set "a" = 0.9 and "the" = 0.05, then wrote that `argmax` always picks
> **"the"**, and that "it" is picked 90% of the time. `argmax` picks **"a"** — the highest
> probability token. The sentence then uses "it" and "the" interchangeably, so the example
> contradicts itself. Reread it and you'll spot it; the underlying understanding is fine, so
> this is a wording error rather than a misconception.
> **The trade-off:** "giving up some consistency" is true but soft. Name the two concrete
> costs:
> 1. **Reproducibility.** The same prompt now yields different text every call, which is why
>    the notebook sets `torch.manual_seed(123)` before sampling. Debugging a stochastic
>    decoder is materially harder.
> 2. **Tail risk.** Every low-probability token retains a nonzero chance, and across 50
>    sampling steps you will eventually draw one. One bad token conditions everything after
>    it, so a single unlucky draw can derail the whole continuation. **This is precisely the
>    problem top-k exists to solve** — which is the link between Q8 and Q10.
> **What's missing on the gain side:** greedy decoding isn't just "less creative", it has a
> specific pathology — it gets trapped in **repetitive loops** ("the same the same the
> same"), because the highest-probability continuation of a phrase is often that phrase
> again. Sampling is the standard escape from that. And greedy is **not** the
> highest-probability *sequence* — it picks the best token at each step independently, which
> is a locally greedy choice and can be globally worse (what beam search addresses).

**9.** Temperature scaling divides the logits by a positive number before the softmax.
Describe the effect of a temperature above 1 versus below 1, and what the distribution
approaches at each extreme.

> A temperature above one results in more a uniform probability distribution. As it grows, the more random/creative the model output becomes. The opposite is true for a temperature below 1. As values get closer to 0, the more "peaky" the distribution becomes. As a result, high probability tokens are chosen more and low probability ones less. Responses therefore become more rigid and predictable.

> **Feedback — ✅ Both directions correct.** Name the two limits exactly, since the question
> asked what the distribution *approaches*:
> - **T → ∞**: uniform over the **entire 50,257-token vocabulary**. Dividing by a huge
>   number drives every logit toward 0, and `softmax` of equal logits is uniform — output
>   becomes pure noise, not "creative".
> - **T → 0**: a **one-hot** distribution on the argmax. So low temperature doesn't merely
>   resemble greedy decoding, it **converges to it exactly**. That's why your `generate`
>   treats `temperature=0.0` as a special case with an `argmax` branch — at literally 0 the
>   division would be by zero.
> The mechanism in one line, worth holding onto: softmax depends only on the **differences**
> between logits, and dividing by `T` **rescales every gap**. `T > 1` shrinks the gaps
> (flatter), `T < 1` stretches them (peakier). Nothing is reordered — the ranking of tokens
> is identical at every temperature, only the gaps between them change.
> One sharpening: "more rigid and predictable" is right, but the failure mode at low `T` is
> the **repetition loop** from Q8 — as `T → 0` you inherit greedy decoding's pathology
> wholesale. Useful framing: temperature doesn't add or remove information, it just decides
> how much of the model's own uncertainty you act on.

**10.** In `generate`, top-k sampling sets the non-selected logits to `-inf` *before* the
softmax, rather than zeroing their probabilities afterwards. Why is that the right place to
do it?

> When computing a probability distribution with softmax, even elements with a value of 0 affect the calculation. Setting the non-selected logits to `-inf` (instead of 0) before softmax ensures that they do not influence the values of the selected logits and that their computed values are 0.

> **Feedback — ✅ You identified the actual trap.** `exp(0) = 1`, so a logit set to **0 is
> not neutral** — it is a *moderate* score that still contributes a full unit of mass to the
> denominator and can even outrank genuinely negative logits. `exp(-inf) = 0` is the only
> value that removes a token exactly, and it's the same trick as the causal mask in Ch. 3 —
> worth noticing it recurring.
> One precision worth having, since it's a nice result: masking to `-inf` before softmax and
> zeroing probabilities after softmax **then renormalizing** are *mathematically identical*
> — a softmax restricted to a subset is just the normalized exponentials of that subset. So
> the argument for doing it before isn't correctness of the distribution, it's that the
> renormalization is **automatic**: the kept probabilities sum to 1 with no extra step, no
> risk of forgetting to divide, and the result composes cleanly with the temperature
> division that follows (`-inf / T` is still `-inf`). Zeroing after softmax *without*
> renormalizing is the genuinely broken version — the probabilities would sum to less than
> 1.
> Also note the mask has to be applied at **every generation step**, inside the loop: the
> top-k set is different for each new token, so there's nothing to precompute.

---

## Summary

**8 solid, 2 needing work.**

| Q | Topic | Mark |
|---|-------|------|
| 1 | Cross-entropy as negative avg log prob | ✅ pipeline right; be concrete about *why* log (underflow, gradients) |
| 2 | Perplexity | ✅ note it's an *effective* count; 48,725 ≈ vocab size is the punchline |
| 3 | Flattening for `cross_entropy` | ✅ fully correct |
| 4 | Validation loss | ✅ add that the *gap* is the diagnostic, and it gives a stopping point |
| 5 | `eval()` vs `no_grad()` | 🟡 mechanisms right, consequences understated — `no_grad` is about **memory** |
| 6 | `zero_grad()` | ✅ add that accumulation is deliberate (gradient accumulation) |
| 7 | Overfitting on "The Verdict" | ✅ quantify it: 163M params / 5,145 tokens |
| 8 | Greedy vs `multinomial` | 🟡 worked example contradicts itself; name tail risk + repetition loops |
| 9 | Temperature | ✅ state the exact limits: uniform over 50,257, and one-hot = argmax |
| 10 | Top-k with `-inf` | ✅ you caught the `exp(0) = 1` trap |

**The two to revisit:**
- **Q5** — the habit to build is reading "what goes wrong" as a resource question as well as
  a correctness one. Forgetting `no_grad` doesn't give wrong numbers, it exhausts your RAM.
- **Q8** — the concept is sound, so just rewrite the example with consistent token names.
  Then connect it forward: sampling's tail risk is *the* motivation for top-k in Q10.

**Pattern across the whole quiz:** your mechanisms are consistently correct, and the gaps
are almost always in quantifying the consequence. You know dropout is disabled, that
gradients accumulate, that the corpus is small — the next level is the numbers that make
each one bite (gigabytes of activations, a `k`× inflated learning rate, 31,000 parameters
per token). That precision is what separates "I followed the chapter" from "I could debug
this when it breaks".
