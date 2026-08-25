# DGPT v1-base

**Hugging Face model repo:** https://huggingface.co/dipan004/DGPT

A 13,049,856-parameter decoder-only Transformer language model, implemented and
trained **entirely from scratch** — manual forward pass, manual backward pass,
manual AdamW optimizer, no autograd — on TinyStories.

**This is a small base language model. It is NOT an instruction-tuned assistant,
it does not follow instructions, it is not aligned/chat-tuned, and it should not
be compared to ChatGPT-class systems.** It generates short, TinyStories-style
children's narratives and nothing else.

---

## Overview

| | |
|---|---|
| Model family | DGPT (**D**ipan **G**enerative **P**re-trained **T**ransformer) |
| Release | DGPT v1-base |
| Parameters | 13,049,856 |
| Architecture | Decoder-only Transformer, Pre-LN, tied embeddings |
| Dataset | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (full corpus) |
| Tokenizer | Locked byte-level BPE, vocab=6000 (`bpe_6000.json`) |
| Final checkpoint | step 5000 |
| Hardware | 1x NVIDIA Tesla T4 (Kaggle), CuPy backend |
| Framework | None — hand-implemented NumPy/CuPy, no PyTorch/JAX/TF |

## Architecture

The architecture is locked and defined in [`configs/v1-base.json`](configs/v1-base.json):

| param | value |
|---|---|
| vocab_size | 6000 |
| block_size (context length) | 256 |
| d_model | 384 |
| n_layer | 6 |
| n_head | 6 |
| head_dim | 64 |
| d_ff | 1536 |
| activation | GELU (tanh approximation, GPT-2 style) |
| normalization | Pre-LN |
| positional embedding | learned |
| lm_head | tied to token embedding, **no** output bias |

**"From scratch" means:** Linear, Embedding, LayerNorm, GELU, causal multi-head
attention, residual connections, cross-entropy, the full model forward pass,
the full model backward pass, and AdamW are all hand-implemented with manually
derived gradients. No `torch.autograd`, no `loss.backward()`, no
`nn.Transformer`/`nn.MultiheadAttention`, no `torch.optim.AdamW`, no pretrained
models, no pretrained tokenizer. CuPy is used strictly as a GPU numerical
execution backend (an array library, like NumPy) — it does not provide
autograd and was never used to compute a gradient. Every layer's backward pass
was independently gradient-checked against finite differences before being
used for training.

This repository's `src/` contains an **inference-only** reimplementation
(forward pass only, NumPy, no CuPy dependency) — see
["Source code" below](#source-code).

## Tokenizer

`bpe_6000.json` is a **locked** byte-level BPE tokenizer: 256 base byte tokens
+ 5,744 learned merges = 6,000 total vocabulary. It was trained once, outside
this repository, and is treated as a fixed external artifact — it is never
retrained, and its merges/token IDs are never modified. `src/tokenizer.py` only
loads and runs it (encode/decode); it has no training method.

## Dataset

Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(`TinyStoriesV2-GPT4-train.txt`), a synthetically generated (GPT-3.5/GPT-4)
short-story corpus with a deliberately small vocabulary, described in
[Eldan & Li, 2023](https://arxiv.org/abs/2305.07759).

- 2,717,495 stories
- 371,525,259 tokens after BPE tokenization (int32, ~1.38 GB token cache)
- TinyStories is distributed under **CDLA-Sharing-1.0**, not by this project —
  see [Licensing](#licensing) below.

## Training procedure

Two-stage training, both stages using the manually implemented forward/backward/AdamW:

1. **Stage 2 (rapid iteration)** — 50,000-story subset, 6,749,327 train tokens,
   batch size 64, 2000 steps, LR 3e-4 with warmup, weight decay 0.01. Used to
   validate the pipeline and gate correctness before spending full-corpus time.
2. **Full-data continuation** — resumed from the Stage 2 step-2000 checkpoint,
   switched to the full 371.5M-token corpus (367,810,007 train / 3,715,252 val
   tokens), continuation LR 3e-5, trained to a **final step of 5000**.

Checkpoints store model parameters, AdamW moments (`m`, `v`) and timestep,
step number, full config, and RNG state, and a full save→load round-trip
(identical logits and loss on the same batch) was verified before relying on
it — see `optimizer_meta`/`rng_meta` in the checkpoint's `__meta__` blob.

Reported final metrics (full-data run, step 5000): train loss ≈ 3.3,
val loss ≈ 3.3–3.4, val perplexity ≈ 27–29 (see the training notebook for the
exact step-by-step log).

## GPU hardware & throughput

- **Hardware:** NVIDIA Tesla T4 (Turing, SM75, 16 GB VRAM), via Kaggle.
- **Backend:** CuPy 14.0.1 (GPU numerical execution only, no autograd).
- **Measured throughput:** ~3,300–3,440 tokens/sec at batch size 64
  (directly measured during Stage 2 benchmarking and full training — not
  extrapolated).
- FlashAttention-2/3-style fused kernels require Ampere+ (SM80+) and are not
  usable on T4; this model uses standard batched CuPy/cuBLAS matmuls for
  attention, which is the throughput reported above.

## Model limitations

- **Not instruction-tuned.** It completes text; it does not follow commands
  or answer questions reliably.
- **Not aligned/chat-tuned.** No RLHF, no DPO, no safety fine-tuning.
- **No broad world knowledge.** TinyStories uses a deliberately small
  vocabulary and simple narrative structure — the model's "knowledge" is
  essentially limited to children's-story tropes and vocabulary.
- **256-token context window.** Long-range coherence is limited.
- **13M parameters.** Small by any modern standard; expect simple, sometimes
  repetitive or logically inconsistent short stories, not sophisticated
  reasoning.
- **English only**, TinyStories-style prose only.

## How to download

The full training checkpoint (`model.npz`) is **not stored directly in this
Git repository** — see [Licensing](#licensing) and the upload instructions
below for why (it's 136+ MB, over GitHub's raw 100 MB file limit, and it
bundles optimizer state you may not want to redistribute long-term). Get it
from one of:

- The GitHub Releases page for this repository (recommended for GitHub), or
- The Hugging Face model repo (recommended for normal use):
  **https://huggingface.co/dipan004/DGPT**

## How to load the model

```bash
pip install -r requirements.txt
```

```python
from src.generate import load_dgpt
from src.tokenizer import BPETokenizer

tok = BPETokenizer("tokenizer/bpe_6000.json")
model, step = load_dgpt("checkpoints/model.npz")
print(f"Loaded DGPT v1-base at step {step}, {tok.vocab_size} vocab")
```

## How to generate text

```bash
python src/generate.py \
    --checkpoint checkpoints/model.npz \
    --tokenizer tokenizer/bpe_6000.json \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 40
```

or from Python:

```python
from src.generate import load_dgpt, generate_text
from src.tokenizer import BPETokenizer

tok = BPETokenizer("tokenizer/bpe_6000.json")
model, _ = load_dgpt("checkpoints/model.npz")

print(generate_text(model, tok, "The little dog was", max_new_tokens=150))
```

## Reproduction instructions

The full training procedure, including the numerical gradient checks and both
training stages, is in
[`notebooks/DGPT_NanoScratchGPT_FINAL.ipynb`](notebooks/) (Kaggle-oriented; needs
a `TinyStories` dataset attachment and the locked `bpe_6000.json` tokenizer
attached as a Kaggle Dataset). To reproduce:

1. Attach the TinyStories dataset and the locked tokenizer dataset in Kaggle.
2. Run sections 1–20 (environment, backend, tokenizer restore, all layers,
   gradient checks) — do not proceed past Section 20 unless every check passes.
3. Run the Stage 2 rapid-iteration training (Section 24 onward) to validate
   the pipeline on the 50k-story subset.
4. Only after Stage 2 confirms "READY for final training", opt in to the full
   run (`RUN_FINAL_TRAINING = True`) for the full-corpus continuation to
   step 5000.
5. Install training-only dependencies from `requirements-training.txt`
   (CuPy build must match your CUDA toolkit).

The tokenizer must never be retrained — always restore the same locked
`bpe_6000.json`.

## Citation

If you reference this project:

```bibtex
@misc{giri2026dgpt,
  author = {Dipan Giri},
  title  = {DGPT v1-base: A From-Scratch Decoder-Only Transformer Trained on TinyStories},
  year   = {2026},
  note   = {NumPy/CuPy implementation with manual forward, backward, and AdamW},
  url    = {https://github.com/Dipan2004/DGPT}
}
```

Also cite the TinyStories dataset/paper if you use it:

```bibtex
@article{eldan2023tinystories,
  title   = {TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author  = {Eldan, Ronen and Li, Yuanzhi},
  journal = {arXiv preprint arXiv:2305.07759},
  year    = {2023}
}
```

## Licensing

This project bundles several things with **different rights holders** — do not
treat them as one blanket license:

| Component | License | Notes |
|---|---|---|
| Code in this repo (`src/`, `scripts/`, `tests/`, `configs/`) | MIT (see [`LICENSE`](LICENSE)) | Written by Dipan Giri. |
| Model weights (`model.npz` / `model_weights.npz`) | **Not established — flagged, not assumed** | Trained from scratch on TinyStories by Dipan Giri. Whether training-derived weights inherit obligations from the training data's license is a genuinely unsettled question and this repo does not resolve it for you. If you plan to redistribute or commercialize the weights, get your own legal read on how CDLA-Sharing-1.0 (TinyStories' license, see below) interacts with a from-scratch model trained on that data. |
| Tokenizer (`bpe_6000.json`) | **Not established — flagged, not assumed** | A statistical merge table derived from tokenizing TinyStories text. Same caveat as the weights. |
| TinyStories dataset | [CDLA-Sharing-1.0](https://huggingface.co/datasets/roneneldan/TinyStories) | Distributed by its authors (Eldan & Li), **not by this project**. This repo does not redistribute the dataset. |
| Third-party dependencies (NumPy, and CuPy for training) | Their own licenses (BSD-3-Clause for NumPy) | Not modified or redistributed by this repo. |

**This is not legal advice.** No license is invented here for the weights or
tokenizer because the source material doesn't establish one; that's flagged
deliberately rather than guessed at.
