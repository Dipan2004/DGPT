# src/tokenizer.py | 96 lines
"""
Loader for the LOCKED byte-level BPE tokenizer (`bpe_6000.json`, vocab=6000).

This tokenizer was trained once, outside this repository, and must never be
retrained or regenerated. This module only deserializes the finished merge
table and performs encode/decode. There is no training method here by design.

On-disk schema (BPT_V1):
    {
        "vocab_size": 6000,
        "merge_order": [
            [[a_id, b_id], new_id],
            ...
        ]
    }
"""

import json


class BPETokenizer:
    def __init__(self, path):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "merge_order" not in data:
            raise KeyError(
                f"Unrecognized tokenizer file at {path}: expected a 'merge_order' key "
                f"(BPT_V1 schema). Found top-level keys: {list(data.keys())}. "
                "Do not attempt to regenerate this tokenizer."
            )

        merge_order = data["merge_order"]

        # Base vocabulary: raw bytes 0..255
        self.id_to_bytes = {i: (i,) for i in range(256)}
        self.bpe_ranks = {}

        for rank, entry in enumerate(merge_order):
            a_id, b_id, new_id = self._normalize_merge_entry(entry)
            self.bpe_ranks[(a_id, b_id)] = rank
            self.id_to_bytes[new_id] = (a_id, b_id)

        self.bytes_to_id = {v: k for k, v in self.id_to_bytes.items()}

        declared_vocab_size = data.get("vocab_size")
        computed_vocab_size = max(self.id_to_bytes.keys()) + 1
        if declared_vocab_size is not None and declared_vocab_size != computed_vocab_size:
            raise ValueError(
                f"Tokenizer mismatch: declared vocab_size={declared_vocab_size}, "
                f"computed={computed_vocab_size}. The tokenizer artifact is locked; "
                "do not regenerate it."
            )
        self.vocab_size = computed_vocab_size

    @staticmethod
    def _normalize_merge_entry(entry):
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], (list, tuple))
            and len(entry[0]) == 2
        ):
            return int(entry[0][0]), int(entry[0][1]), int(entry[1])
        raise KeyError(f"Unrecognized merge entry: {entry!r}")

    def _get_pairs(self, seq):
        return set(zip(seq[:-1], seq[1:]))

    def _bpe_merge(self, seq):
        seq = list(seq)
        if len(seq) < 2:
            return seq
        while True:
            pairs = self._get_pairs(seq)
            ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
            if not ranked:
                break
            _, best = min(ranked)
            new_seq, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                    new_seq.append(self.bytes_to_id[(seq[i], seq[i + 1])])
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        return seq

    def encode(self, text):
        raw_bytes = text.encode("utf-8")
        return self._bpe_merge(list(raw_bytes))

    def _expand(self, token_id):
        if token_id < 256:
            return bytes([token_id])
        out = bytearray()
        for part in self.id_to_bytes[token_id]:
            out.extend(self._expand(part) if part >= 256 else bytes([part]))
        return bytes(out)

    def decode(self, ids):
        out = bytearray()
        for token_id in ids:
            out.extend(self._expand(int(token_id)))
        return bytes(out).decode("utf-8", errors="replace")
