# scripts/extract_weights.py | 45 lines
"""
Extracts an inference-only weights file from the full training checkpoint.

The full checkpoint (model.npz, i.e. the renamed dgpt_full_step5000.npz)
contains model parameters (`param__*`) PLUS AdamW optimizer state
(`m__*`, `v__*`) and a JSON `__meta__` blob (config, RNG state, optimizer
timestep). That optimizer state is only needed to resume training and is
not needed for inference.

This script copies ONLY the `param__*` arrays into a new, much smaller
`model_weights.npz`, for people who just want to run generation without
downloading optimizer state.

This does NOT modify or overwrite the original checkpoint.

Usage:
    python scripts/extract_weights.py \
        --in checkpoints/model.npz \
        --out checkpoints/model_weights.npz
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                         help="Path to the full checkpoint (model.npz)")
    parser.add_argument("--out", dest="out_path", required=True,
                         help="Output path for the inference-only weights file")
    args = parser.parse_args()

    data = np.load(args.in_path, allow_pickle=True)
    param_keys = [k for k in data.files if k.startswith("param__")]
    if not param_keys:
        raise ValueError(f"No 'param__*' arrays found in {args.in_path}")

    weights = {k: data[k] for k in param_keys}
    np.savez_compressed(args.out_path, **weights)

    total_params = sum(v.size for v in weights.values())
    print(f"Extracted {len(weights)} parameter arrays ({total_params:,} total params)")
    print(f"Wrote inference-only weights to: {args.out_path}")
    print(f"Original checkpoint left untouched at: {args.in_path}")


if __name__ == "__main__":
    main()
