# Quiz — Ch. 2: Tokenization & data loading

Self-test on the Ch. 2 material. Attempt each cold, then ask me for feedback once
you've written your answers. Mix of recall, "why", and code-reasoning questions.

---

**1.** Why does GPT-2's BPE tokenizer never need an `<|unk|>` token, while
`SimpleTokenizerV2` does?

**2.** In `GPTDatasetV1`, the target chunk is `token_ids[i+1 : i+max_length+1]`. In one
sentence, what training objective does that shift encode?

**3.** You set `max_length=256` and `stride=256`. Then you change to `stride=128`. What
changes about the resulting dataset, and what's the risk?

**4.** What are the two embedding tables in a GPT input layer, and why is the second one
necessary?

**5.** GPT-2's vocab size is 50,257. Which token is ID 50256, and what is it used for?

**6.** In one sentence each, what do `batch_size`, `shuffle`, and `drop_last` control in
`create_dataloader_v1`?

**7.** What is the shape of a single batch produced by `create_dataloader_v1`, and what
do its dimensions mean?
