"""Compute per-dimension std of GT shape (45D) and selected scale (10D) parameters
across all MHR-format training NPZ files. Saves results to a .pt file used by
NFARHead to set per-dimension perturbation scales for mode-2 training.

Usage:
    python tools/compute_shape_scale_std.py [--out PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sam_3d_body.configs.config import DATASET_FILES

SCALE_INDICES = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]

DEFAULT_OUT = "checkpoints/sam-3d-body-dinov3/shape_scale_std.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output .pt file path (default: %(default)s)")
    args = parser.parse_args()

    # Only use training NPZ files that contain MHR parameters (identity_coeffs key).
    # Real-world datasets (coco, mpii, h36m, …) use a different format.
    train_files = {
        name: path for name, path in DATASET_FILES[1].items()
        if "all_npz_12_training_mhr_conditioned" in path
    }

    print(f"Loading {len(train_files)} MHR training files…")

    shape_chunks = []
    scale_chunks = []

    for name, path in train_files.items():
        p = Path(path)
        if not p.exists():
            print(f"  SKIP (missing): {name}")
            continue
        data = np.load(path, allow_pickle=True)
        if "identity_coeffs" not in data:
            print(f"  SKIP (no identity_coeffs): {name}")
            continue
        shape = data["identity_coeffs"].astype(np.float32)   # (N, 45)
        scale = data["lbs_model_params"][:, -68:].astype(np.float32)  # (N, 68)
        scale_sel = scale[:, SCALE_INDICES]                  # (N, 10)
        shape_chunks.append(shape)
        scale_chunks.append(scale_sel)
        print(f"  {name:40s}  n={len(shape):6d}")

    all_shape = np.concatenate(shape_chunks, axis=0)   # (N_total, 45)
    all_scale = np.concatenate(scale_chunks, axis=0)   # (N_total, 10)

    print(f"\nTotal samples: {len(all_shape):,}")

    shape_std = all_shape.std(axis=0)   # (45,)
    scale_std = all_scale.std(axis=0)   # (10,)

    print(f"\nShape std  — min={shape_std.min():.4f}  max={shape_std.max():.4f}  mean={shape_std.mean():.4f}")
    print(f"Scale std  — min={scale_std.min():.4f}  max={scale_std.max():.4f}  mean={scale_std.mean():.4f}")
    print(f"\nShape std per dim:\n{np.round(shape_std, 4)}")
    print(f"\nScale std per dim (indices {SCALE_INDICES}):\n{np.round(scale_std, 4)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "shape_std": torch.tensor(shape_std),
            "scale_std": torch.tensor(scale_std),
            "scale_indices": SCALE_INDICES,
        },
        out,
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
