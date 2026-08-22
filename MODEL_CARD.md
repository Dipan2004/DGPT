---
license: unknown
language:
- en
tags:
- text-generation
- from-scratch
- transformer
- gpt
- numpy
- cupy
- tinystories
datasets:
- roneneldan/TinyStories
pipeline_tag: text-generation
---

# DGPT v1-base

**A small, from-scratch base language model. Not an instruction-tuned assistant.**

DGPT v1-base is a 13,049,856-parameter decoder-only Transformer trained from
scratch (manual forward pass, manual backward pass, manual AdamW — no
autograd, no PyTorch/JAX/TensorFlow) on the TinyStories dataset. It generates
short, simple, TinyStories-style children's narratives and nothing more.

**Do not expect:** instruction following, multi-turn conversation, reasoning,
factual world knowledge, or ChatGPT-comparable capability of any kind. This
model was never trained or tuned for any of those.

## Model description

- **Model type:** decoder-only Transformer, Pre-LN, GELU (tanh approx), tied
  token embedding / LM head (no output bias), learned positional embeddings.
- **Parameters:** 13,049,856
- **Context length:** 256 tokens
- **Vocabulary:** 6,000 (locked byte-level BPE, `bpe_6000.json`)
- **Framework:** none — hand-implemented NumPy/CuPy. Every layer's backward
  pass was independently verified against finite-difference gradient checks
  before training.

## Architecture

| param | value |
|---|---|
| vocab_size | 6000 |
| block_size | 256 |
| d_model | 384 |
| n_layer | 6 |
| n_head | 6 |
| head_dim | 64 |
| d_ff | 1536 |
| activation | GELU (tanh approx) |
| norm | Pre-LN |
| positions | learned |
| lm_head | tied to token embedding, no bias |

## Training data

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(`TinyStoriesV2-GPT4-train.txt`): 2,717,495 synthetically generated (GPT-3.5/
GPT-4) short stories using a deliberately small vocabulary
([Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)), tokenized to
371,525,259 tokens. Licensed by its authors under CDLA-Sharing-1.0 — this
model card does not redistribute the dataset itself.

## Training procedure

- **Optimizer:** manually implemented AdamW (lr, betas, weight decay applied
  only to matrix params — biases/LayerNorm params excluded from decay).
- **Stage 2 (validation run):** 50k-story subset, 2000 steps, batch size 64,
  LR 3e-4 with warmup, used to gate correctness before full training.
- **Full run:** resumed from the Stage 2 checkpoint, continued on the full
  371.5M-token corpus at LR 3e-5, to a **final checkpoint at step 5000**.
- **Tokenizer:** locked, pre-trained externally, never retrained during model
  training.

## Hardware

- 1x NVIDIA Tesla T4 (Turing, SM75, 16 GB VRAM), Kaggle.
- CuPy 14.0.1 as the GPU numerical execution backend (no autograd usage).
- Measured throughput: ~3,300–3,440 tokens/sec at batch size 64 (directly
  measured, not extrapolated).

## Intended use

- Educational reference for from-scratch Transformer implementation
  (manual forward/backward/AdamW) at small scale.
- Generating short, TinyStories-style children's narratives from a prompt.
- Portfolio / ML-engineering demonstration.

## Out-of-scope use

- Any production or consumer-facing assistant use case.
- Instruction following, chat, question answering, factual retrieval,
  reasoning tasks, or code generation.
- Anything requiring broad world knowledge — the model's effective knowledge
  is bounded by TinyStories' simplified vocabulary and narrative style.
- Any use that assumes safety alignment or content filtering — **none was
  performed.**

## Evaluation

From the training notebook (full-data run, step 5000):

- Train loss ≈ 3.3
- Val loss ≈ 3.3–3.4
- Val perplexity ≈ 27–29

No held-out benchmark suite (e.g. downstream NLP tasks) was run — TinyStories
train/val loss and perplexity are the only reported metrics. Treat any
numbers as approximate; see the training notebook's step-by-step log for the
exact source values.

## Known generation issues

- Occasional run-on or abruptly concatenated sentences (short stories
  sometimes blend into the next without a clean boundary).
- Repetition of simple phrases/character names across generations.
- No factual grounding — names, objects, and events are generated freely and
  are not to be treated as accurate about anything.
- Context is capped at 256 tokens; longer prompts are truncated from the
  left before generation.

## How to use

```python
from src.generate import load_dgpt, generate_text
from src.tokenizer import BPETokenizer

tok = BPETokenizer("tokenizer/bpe_6000.json")
model, _ = load_dgpt("model.npz")

print(generate_text(model, tok, "Once upon a time", max_new_tokens=150))
```

## Licensing

License is marked `unknown` above deliberately. See this repository's main
`README.md` → "Licensing" for the full breakdown across code, weights,
tokenizer, and the TinyStories dataset — the weights and tokenizer do not
have an established license and none is invented here.
