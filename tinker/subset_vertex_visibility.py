"""Pre-compute per-shard dense-keypoint visibility from the 18,439-wide full
vertex visibility files.

For every `<shard>_vertex_visibility.npz` in the visibility-labels dir, write a
companion `<shard>_selected_vertex_visibility.npz` containing just the
visibility flags for the `MHR_DENSE_KP_INDICES` vertices — shape `(N, K)`,
`bool`. The training dataset loads these lightweight files directly, avoiding
the repeated 18,439-wide read + subset at dataset-init time.

Idempotent: skips shards that already have the selected file (pass
`--overwrite` to regenerate, e.g. if `MHR_DENSE_KP_INDICES` changes).

Usage:
    python tinker/subset_vertex_visibility.py
    python tinker/subset_vertex_visibility.py --overwrite
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sam_3d_body.data.bedlam_dataset import MHR_DENSE_KP_INDICES  # noqa: E402


VIS_DIR = "/scratch/cq244/BEDLAM/data/training_labels/visibility_labels"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_dir", default=VIS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    idx = np.asarray(MHR_DENSE_KP_INDICES)
    K = len(idx)
    print(f"Subsetting to {K} vertices; scanning {args.vis_dir}")

    files = sorted(
        f for f in os.listdir(args.vis_dir) if f.endswith("_vertex_visibility.npz")
    )
    print(f"Found {len(files)} vertex-visibility files")

    for name in files:
        src = os.path.join(args.vis_dir, name)
        dst = os.path.join(args.vis_dir, name.replace(
            "_vertex_visibility.npz", "_selected_vertex_visibility.npz"
        ))
        if os.path.exists(dst) and not args.overwrite:
            print(f"[skip] {os.path.basename(dst)} exists")
            continue

        full = np.load(src)["vertex_visibility"]              # (N, 18439) bool
        selected = full[:, idx].astype(bool)                  # (N, K) bool
        np.savez_compressed(dst, vertex_visibility=selected)
        print(f"[ok]   {os.path.basename(dst)}  shape={selected.shape} mean={selected.mean():.3f}")


if __name__ == "__main__":
    main()
