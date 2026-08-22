# src/model.py | 158 lines
"""
DGPT — from-scratch decoder-only Transformer (inference-only reimplementation).

This is a forward-pass-only extraction of the architecture defined in the
original training notebook (NanoScratchGPT / DGPT). The original notebook
also hand-implements the backward pass and an AdamW optimizer for training;
those are intentionally NOT reproduced here since this file is for loading
`model.npz` and generating text only.

Locked architecture (see configs/v1-base.json):
    vocab_size = 6000
    block_size = 256
    d_model    = 384
    n_layer    = 6
    n_head     = 6
    head_dim   = 64
    d_ff       = 1536
    activation = GELU (tanh approximation, GPT-2 style)
    norm       = Pre-LN
    positions  = learned
    lm_head    = tied to token embedding, no output bias

No PyTorch, no Hugging Face Transformers. NumPy only.
"""

import numpy as np


def stable_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    c = np.float32((2.0 / np.pi) ** 0.5)
    inner = c * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


class Linear:
    def __init__(self, W, b=None):
        self.W = W
        self.b = b

    def __call__(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out


class LayerNorm:
    def __init__(self, gamma, beta, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class CausalSelfAttention:
    def __init__(self, params, prefix, n_head):
        self.Wq = Linear(params[f"{prefix}.Wq.W"], params[f"{prefix}.Wq.b"])
        self.Wk = Linear(params[f"{prefix}.Wk.W"], params[f"{prefix}.Wk.b"])
        self.Wv = Linear(params[f"{prefix}.Wv.W"], params[f"{prefix}.Wv.b"])
        self.Wo = Linear(params[f"{prefix}.Wo.W"], params[f"{prefix}.Wo.b"])
        self.n_head = n_head

    def __call__(self, x):
        B, T, C = x.shape
        H = self.n_head
        hd = C // H

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        def split_heads(t):
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        scale = np.float32(1.0 / (hd ** 0.5))
        scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) * scale

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, np.float32(-1e9), scores)

        A = stable_softmax(scores, axis=-1)
        ctx = np.matmul(A, Vh)
        ctx_merged = ctx.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.Wo(ctx_merged)


class FeedForward:
    def __init__(self, params, prefix):
        self.fc1 = Linear(params[f"{prefix}.fc1.W"], params[f"{prefix}.fc1.b"])
        self.fc2 = Linear(params[f"{prefix}.fc2.W"], params[f"{prefix}.fc2.b"])

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock:
    def __init__(self, params, prefix, n_head):
        self.ln1 = LayerNorm(params[f"{prefix}.ln1.gamma"], params[f"{prefix}.ln1.beta"])
        self.attn = CausalSelfAttention(params, f"{prefix}.attn", n_head)
        self.ln2 = LayerNorm(params[f"{prefix}.ln2.gamma"], params[f"{prefix}.ln2.beta"])
        self.ffn = FeedForward(params, f"{prefix}.ffn")

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DGPT:
    """Inference-only DGPT. Load parameters with `DGPT.from_params(params, config)`."""

    def __init__(self, params, config):
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.block_size = config["block_size"]
        self.d_model = config["d_model"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]

        self.tok_emb_W = params["tok_emb.W"]
        self.pos_emb_W = params["pos_emb.W"]
        self.blocks = [
            TransformerBlock(params, f"blocks.{i}", self.n_head)
            for i in range(self.n_layer)
        ]
        self.ln_f = LayerNorm(params["ln_f.gamma"], params["ln_f.beta"])

    def num_parameters(self):
        return sum(v.size for v in self.__dict__.get("_raw_params", {}).values())

    def forward(self, idx):
        """idx: int array (B, T) with T <= block_size. Returns logits (B, T, vocab_size)."""
        B, T = idx.shape
        assert T <= self.block_size, "sequence length exceeds block_size"

        tok = self.tok_emb_W[idx]
        pos = self.pos_emb_W[:T]
        x = tok + pos[None, :, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = x @ self.tok_emb_W.T  # tied weights, no output bias
        return logits
