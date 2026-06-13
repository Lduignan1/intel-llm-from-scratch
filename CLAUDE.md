# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-notebook learning project that implements a GPT-style LLM from scratch, following Sebastian Raschka's *Build a Large Language Model From Scratch*, targeting Intel hardware. Nearly all work happens in `llm_from_scratch.ipynb` (~194 cells). There are no `.py` source files — code is authored and run as notebook cells.

## Environment & commands

Dependencies are managed with `uv` (`pyproject.toml`, `uv.lock`, Python pinned to 3.10 in `.python-version`).

- `uv sync` — install declared dependencies into `.venv`
- `uv run jupyter lab` (or `jupyter notebook`) — launch the notebook
- `uv run jupyter nbconvert --to notebook --execute --inplace llm_from_scratch.ipynb` — run the whole notebook end-to-end

There is no test suite, linter, or build step.

**Important dependency gap:** the notebook imports `torch`, `tiktoken`, and `matplotlib`, but `pyproject.toml` only declares `jupyter`. These ML libraries are not in `uv.lock` and must be installed separately (e.g. `uv pip install torch tiktoken matplotlib`). When adding code that needs a new library, expect to install it manually rather than assuming `uv sync` provides it.

## Notebook structure

The notebook builds the model incrementally, mirroring the book's chapters. Sections (by markdown header):

- **Ch. 2 — Working with text data**: regex tokenizer (`basic_tokenizer`), `SimpleTokenizerV1`/`V2` (vocab + `<|unk|>`/`<|endoftext|>` special tokens), byte-pair encoding via `tiktoken`, then `GPTDatasetV1` + `create_dataloader_v1` for sliding-window input/target pairs.
- **Ch. 3 — Attention**: progression of `SelfAttention_v1` → `SelfAttention_v2` (uses `nn.Linear`) → `CausalAttention` (masking + dropout) → `MultiHeadAttentionWrapper` (stacked heads) → `MultiHeadAttention` (single matrix with weight splits). Each class is a refinement of the previous — preserve this teaching progression rather than collapsing them.
- **Ch. 4 — GPT model**: `DummyGPTModel`/`DummyTransformerBlock`/`DummyLayerNorm` placeholders, then real `LayerNorm`, `GELU`, and `FeedForward` implementations.

`the-verdict.txt` (Edith Wharton, "The Verdict") is the text corpus used throughout for tokenization and dataloader examples.

## Conventions

- Versioned class names (`...V1`, `...V2`, `_v1`, `_v2`) are intentional — they show successive implementations of the same concept. Don't rename or merge them.
- Code targets Intel hardware; when adding device/accelerator code, prefer Intel paths (`torch.xpu`, but `torch.cuda` should also be an option as sometimes use Colab) and write device-agnostic checks rather than hardcoding CUDA.
- when creating quizzes, focus on concepts learned rather than just the code that is present in the notebook. Some pytorch questions can and should be asked, but only for the important elements. 
