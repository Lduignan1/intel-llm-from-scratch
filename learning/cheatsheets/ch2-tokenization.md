# Ch. 2 — Tokenization & data loading

> Maps to: `basic_tokenizer`, `SimpleTokenizerV1/V2`, BPE via `tiktoken`,
> `GPTDatasetV1`, `create_dataloader_v1` in the notebook.

## The pipeline at a glance

```
raw text → tokens → token IDs → (sliding window) → input/target pairs → batches → embeddings
```

A language model never sees text. It sees **integer IDs**, which it turns into
**vectors** (embeddings). Everything in Ch. 2 is about getting from a `.txt` file
to batched (input, target) tensors of IDs.

## Stages

### 1. Tokenization — splitting text into units

- **Word/regex level** (`basic_tokenizer`): `re.split(r'([,.:;?_!"()\']|--|\s)', text)`.
  Simple, but vocabulary explodes and unknown words break it.
- **Subword / BPE** (Byte-Pair Encoding, what GPT-2 actually uses, via
  `tiktoken.get_encoding("gpt2")`): starts from bytes and greedily merges the most
  frequent adjacent pairs into new tokens. Why it wins:
  - **No `<unk>` ever** — any string is representable, worst case falls back to bytes.
  - Fixed vocab size (GPT-2: **50,257**) regardless of how big the corpus is.
  - Common words = 1 token; rare words = a few subword pieces.

### 2. Vocabulary & special tokens

- Vocab = a `{token: id}` dict. `SimpleTokenizerV1` builds it from the corpus.
- `SimpleTokenizerV2` adds **special tokens**:
  - `<|unk|>` — placeholder for out-of-vocab tokens (the BPE tokenizer doesn't need this).
  - `<|endoftext|>` — document separator / sequence boundary. In GPT-2's BPE this is
    ID **50256** (the last ID).

### 3. Sliding window → input/target pairs (`GPTDatasetV1`)

The training signal for an LLM is **next-token prediction**. The target is just the
input shifted right by one:

```python
for i in range(0, len(token_ids) - max_length, stride):
    input_chunk  = token_ids[i      : i + max_length]      # x
    target_chunk = token_ids[i + 1  : i + max_length + 1]  # y = x shifted by 1
```

- `max_length` (context length): how many tokens the model sees at once.
- `stride`: how far the window jumps between samples.
  - `stride == max_length` → no overlap between samples (no token reuse).
  - `stride < max_length` → overlapping windows, more samples, risk of overfitting on
    repeated spans.

### 4. Batching (`create_dataloader_v1`)

Wraps the dataset in a PyTorch `DataLoader`:
- `batch_size` — samples processed together (gradient is averaged over them).
- `shuffle=True` — randomize sample order each epoch (training); `False` for inspection.
- `drop_last=True` — drop a final undersized batch so all batches have equal shape.
- `num_workers` — parallel data-loading processes (0 = main process).

A batch has shape `[batch_size, max_length]` of token IDs.

## Embeddings (the bridge to Ch. 3)

Two embedding tables, **added together**:
1. **Token embedding**: `nn.Embedding(vocab_size, emb_dim)` — a learned vector per token ID.
2. **Positional embedding**: `nn.Embedding(context_length, emb_dim)` — a learned vector
   per position. Needed because attention itself is order-agnostic (see Ch. 3 cheatsheet).

`input_embeddings = token_emb(ids) + pos_emb(positions)` → shape
`[batch, max_length, emb_dim]`. This tensor is what attention consumes.

## Common pitfalls / things to verify

- An `nn.Embedding` lookup is **not** a matrix multiply by a one-hot vector
  conceptually, but it's mathematically equivalent — it's just an indexed row fetch.
- `target` is shifted by **1**, not by `max_length`. Each position predicts *the very
  next* token.
- Vocab size and the token-embedding table's first dim must match (50257 for GPT-2 BPE).

## Key terms

| Term | One-liner |
|------|-----------|
| Token | Atomic unit the model sees (subword for BPE) |
| Vocabulary | token ↔ id mapping; size fixes the embedding/output dims |
| BPE | Subword tokenizer built by merging frequent byte pairs |
| Context length | Max tokens per sample (`max_length`) |
| Stride | Step between sliding windows |
| Embedding | Learned vector for a token ID (or position) |
