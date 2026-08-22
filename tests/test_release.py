# tests/test_release.py | 96 lines
"""
Lightweight release verification tests. No training/backward, no CuPy.

Run from the repo root:
    python -m pytest tests/test_release.py -v

Requires (relative to repo root, as noted in README):
    checkpoints/model.npz
    tokenizer/bpe_6000.json
"""

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tokenizer import BPETokenizer  # noqa: E402
from generate import load_dgpt, generate_text  # noqa: E402

TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizer", "bpe_6000.json")
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints", "model.npz")
EXPECTED_PARAM_COUNT = 13_049_856
EXPECTED_VOCAB_SIZE = 6000
EXPECTED_FINAL_STEP = 5000

requires_tokenizer = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH),
    reason="tokenizer/bpe_6000.json not present (add it manually, see README)",
)
requires_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH),
    reason="checkpoints/model.npz not present (download it, see README)",
)


@requires_tokenizer
def test_tokenizer_loads():
    tok = BPETokenizer(TOKENIZER_PATH)
    assert tok is not None


@requires_tokenizer
def test_vocab_size():
    tok = BPETokenizer(TOKENIZER_PATH)
    assert tok.vocab_size == EXPECTED_VOCAB_SIZE


@requires_tokenizer
def test_encode_decode_roundtrip():
    tok = BPETokenizer(TOKENIZER_PATH)
    text = "Once upon a time, there was a little dog."
    ids = tok.encode(text)
    assert isinstance(ids, list) and len(ids) > 0
    decoded = tok.decode(ids)
    assert decoded == text


@requires_checkpoint
def test_checkpoint_loads():
    model, step = load_dgpt(CHECKPOINT_PATH)
    assert model is not None


@requires_checkpoint
def test_parameter_count():
    model, _ = load_dgpt(CHECKPOINT_PATH)
    total = sum(v.size for v in {
        "tok_emb.W": model.tok_emb_W, "pos_emb.W": model.pos_emb_W,
    }.values())
    # Full count via the raw npz, since DGPT doesn't retain a flat param dict.
    data = np.load(CHECKPOINT_PATH, allow_pickle=True)
    param_keys = [k for k in data.files if k.startswith("param__")]
    total = sum(data[k].size for k in param_keys)
    assert total == EXPECTED_PARAM_COUNT


@requires_checkpoint
def test_final_checkpoint_step():
    data = np.load(CHECKPOINT_PATH, allow_pickle=True)
    if "__meta__" not in data.files:
        pytest.skip("weights-only npz has no step metadata")
    meta = json.loads(str(data["__meta__"]))
    assert meta["step"] == EXPECTED_FINAL_STEP


@requires_checkpoint
@requires_tokenizer
def test_generation_produces_text():
    tok = BPETokenizer(TOKENIZER_PATH)
    model, _ = load_dgpt(CHECKPOINT_PATH)
    output = generate_text(model, tok, "Once upon a time", max_new_tokens=20, seed=0)
    assert isinstance(output, str)
    assert len(output) > 0
