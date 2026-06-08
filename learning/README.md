# Learning resources

Teaching material for working through *Build a Large Language Model From Scratch*
in `llm_from_scratch.ipynb`. Everything here is tied to the code **you** wrote in
the notebook — same class names, same tensor shapes — so the theory maps directly
onto your implementation.

## How to use

- **Cheatsheets** (`cheatsheets/`) — dense, scannable references for each topic.
  Read after implementing a section; use them to consolidate and to look things up.
- **Quizzes** (`quizzes/`) — one per chapter, questions only (no answers). Attempt
  them cold, write your answers, then ask me for feedback and I'll mark them.

## Contents

| Topic | Cheatsheet | Quiz |
|-------|-----------|------|
| Ch. 2 — Tokenization & data loading | [ch2-tokenization.md](cheatsheets/ch2-tokenization.md) | [ch2-quiz.md](quizzes/ch2-quiz.md) |
| Ch. 3 — Attention | [ch3-attention.md](cheatsheets/ch3-attention.md) | [ch3-quiz.md](quizzes/ch3-quiz.md) |
| Ch. 4 — GPT components | [ch4-gpt-components.md](cheatsheets/ch4-gpt-components.md) | [ch4-quiz.md](quizzes/ch4-quiz.md) |

## Coverage status

- [x] Ch. 2 — Working with text data
- [x] Ch. 3 — Attention mechanisms
- [x] Ch. 4 — GPT model components (LayerNorm, GELU, FeedForward)
- [ ] Ch. 4 — Full transformer block + GPT assembly *(in progress in notebook)*
- [ ] Ch. 5 — Pretraining / training loop
- [ ] Ch. 6 — Fine-tuning
