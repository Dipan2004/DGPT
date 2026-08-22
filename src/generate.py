# src/generate.py | 158 lines
"""
DGPT v1-base inference: load model.npz + bpe_6000.json, generate text.

Usage:
    python src/generate.py \
        --checkpoint checkpoints/model.npz \
        --tokenizer tokenizer/bpe_6000.json \
        --prompt "Once upon a time" \
        --max_new_tokens 200 --temperature 0.8 --top_k 40

Accepts either:
  - the full training checkpoint (model.npz), which contains
    `param__*`, `m__*`, `v__*`, and a JSON `__meta__` blob with the model
    config and optimizer state, OR
  - an inference-only weights file (model_weights.npz) produced by
    scripts/extract_weights.py, which contains ONLY `param__*` keys and
    requires --config to be passed explicitly (defaults to
    configs/v1-base.json).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DGPT, stable_softmax  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "v1-base.json"
)


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return json.load(f)


def load_dgpt(checkpoint_path, config_path=None):
    """
    Loads a DGPT model for inference from either a full training checkpoint
    or an inference-only weights file. Returns (model, step_or_none).
    """
    data = np.load(checkpoint_path, allow_pickle=True)
    param_keys = [k for k in data.files if k.startswith("param__")]
    if not param_keys:
        raise ValueError(f"No 'param__*' arrays found in {checkpoint_path}")

    params = {k[len("param__"):]: data[k] for k in param_keys}

    if "__meta__" in data.files:
        meta = json.loads(str(data["__meta__"]))
        config = meta["config"]
        step = meta.get("step")
    else:
        config = load_config(config_path)
        step = None

    model = DGPT(params, config)
    return model, step


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40,
                   seed=None):
    """Autoregressive sampling. NumPy only (CPU inference)."""
    rng = np.random.default_rng(seed)
    generated = list(tokenizer.encode(prompt))

    for _ in range(max_new_tokens):
        context = generated[-model.block_size:]
        idx = np.asarray([context], dtype=np.int64)

        logits = model.forward(idx)
        logits_last = logits[0, -1].astype(np.float32, copy=False)

        temperature = max(float(temperature), 1e-6)
        logits_last = logits_last / temperature

        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits_last.shape[0])
            top_idx = np.argpartition(logits_last, -k)[-k:]
            filtered = np.full_like(logits_last, -1e10)
            filtered[top_idx] = logits_last[top_idx]
            logits_last = filtered

        probs = stable_softmax(logits_last, axis=-1)
        next_token = int(rng.choice(probs.shape[0], p=probs / probs.sum()))
        generated.append(next_token)

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="DGPT v1-base text generation")
    parser.add_argument("--checkpoint", default="checkpoints/model.npz")
    parser.add_argument("--tokenizer", default="tokenizer/bpe_6000.json")
    parser.add_argument("--config", default=None, help="Only needed for weights-only npz files")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer} ...")
    tok = BPETokenizer(args.tokenizer)
    assert tok.vocab_size == 6000, f"Expected vocab_size=6000, got {tok.vocab_size}"

    print(f"Loading model from {args.checkpoint} ...")
    model, step = load_dgpt(args.checkpoint, args.config)
    print(f"Model loaded. step={step} vocab={model.vocab_size} block_size={model.block_size}")

    print(f"\nPrompt: {args.prompt!r}\n")
    output = generate_text(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print("Output:")
    print(output)


if __name__ == "__main__":
    main()
