"""
Visualise BEDLAM ground-truth quality: SMPL-X vs converted MHR (before/after).

For a handful of frames from the closeup_suburb_b sequences, renders:
  - SMPL-X GT (posed + neutral)
  - MHR from all_npz_12_training_mhr_fixed (posed + neutral)
  - MHR from all_npz_12_training_mhr_conditioned, if present (posed + neutral)
  - Side-by-side overlays for each pairing

Output: one PNG per sequence × {posed, neutral} saved to --output_dir.

Usage:
    python scripts/visualise_bedlam_gt.py
    python scripts/visualise_bedlam_gt.py --n_frames 6 --output_dir vis_gt --gpu 0
"""

import os
import sys
import argparse
import numpy as np
import torch
import smplx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mhr.mhr import MHR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MHR_ASSETS_PATH     = "/scratch/cq244/MHR/assets"
MHR_MODEL_FILES_PATH = "/scratch/cq244/MHR/model_files"

FIXED_PATH       = "/scratch/cq244/BEDLAM/data/training_labels/all_npz_12_training_mhr_fixed"
CONDITIONED_PATH = "/scratch/cq244/BEDLAM/data/training_labels/all_npz_12_training_mhr_conditioned"
EXTRA_PATH       = "/scratch/cq244/BEDLAM/data/training_labels/all_npz_12_training_extra"

SUBURB_B_FILES = [
    "20221011_1_250_batch01hand_closeup_suburb_b_6fps.npz",
    # "20221019_1_250_highbmihand_closeup_suburb_b_6fps.npz",
]

# lbs_model_params layout: [transls(3), rots(3), pose_params(130), scale_params(68)]
_LBS_TRANSL_END   = 3
_LBS_ROTS_END     = 6
_LBS_POSE_END     = 136   # 6 + 130
_LBS_SCALE_START  = 136


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scatter3d(ax, verts, color=None, cmap="viridis", s=0.4, alpha=0.5, label=None,
               stride=8):
    """Plot every stride-th vertex as a 3-D scatter."""
    v = verts[::stride]
    c = color if color is not None else v[:, 1]
    ax.scatter(v[:, 0], v[:, 2], v[:, 1],
               c=c, cmap=cmap if color is None else None,
               s=s, alpha=alpha, label=label, rasterized=True)


def _set_equal_axes(ax, verts):
    extent = max((verts.max(0) - verts.min(0))) / 2
    mid    = (verts.max(0) + verts.min(0)) / 2
    ax.set_xlim(mid[0] - extent, mid[0] + extent)
    ax.set_ylim(mid[2] - extent, mid[2] + extent)
    ax.set_zlim(mid[1] - extent, mid[1] + extent)
    ax.view_init(elev=10, azim=-80)
    ax.set_axis_off()


def _neutral_lbs(lbs: np.ndarray) -> np.ndarray:
    """Return lbs_model_params with pose zeroed (keep scale; zero transl/rots/pose)."""
    out = np.zeros_like(lbs)
    out[..., _LBS_SCALE_START:] = lbs[..., _LBS_SCALE_START:]
    return out


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class GTVisualizer:
    def __init__(self, device: torch.device):
        self._device = device

        self._smplx = smplx.create(
            model_type="smplx",
            model_path=MHR_MODEL_FILES_PATH,
            num_betas=11,
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

        self._mhr = MHR.from_files(folder=Path(MHR_ASSETS_PATH), lod=1, device=device)

    # ---- SMPL-X forward ----

    @torch.no_grad()
    def smplx_vertices(self, data: dict, indices: np.ndarray,
                       neutral: bool = False) -> np.ndarray:
        """Return (N, V, 3) SMPL-X vertices in metres."""
        B = len(indices)
        zero3  = torch.zeros(B, 3,  device=self._device)
        zero63 = torch.zeros(B, 63, device=self._device)
        zero45 = torch.zeros(B, 45, device=self._device)

        betas        = torch.tensor(data["shape"][indices],    dtype=torch.float32, device=self._device)
        global_orient = (zero3  if neutral else
                         torch.tensor(data["pose_cam"][indices, :3], dtype=torch.float32, device=self._device))
        body_pose     = (zero63 if neutral else
                         torch.tensor(data["pose_cam"][indices, 3:66], dtype=torch.float32, device=self._device))
        left_hand     = (zero45 if neutral else
                         torch.tensor(data["pose_cam"][indices, 75:120], dtype=torch.float32, device=self._device))
        right_hand    = (zero45 if neutral else
                         torch.tensor(data["pose_cam"][indices, 120:165], dtype=torch.float32, device=self._device))

        out = self._smplx(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            jaw_pose=zero3,
            leye_pose=zero3,
            reye_pose=zero3,
            expression=torch.zeros(B, self._smplx.num_expression_coeffs, device=self._device),
        )
        return out.vertices.cpu().numpy()  # (B, V, 3) metres

    # ---- MHR forward ----

    @torch.no_grad()
    def mhr_vertices(self, data: dict, indices: np.ndarray,
                     neutral: bool = False) -> np.ndarray:
        """Return (N, V, 3) MHR vertices in centimetres."""
        lbs  = data["lbs_model_params"][indices].copy()   # (B, 204)
        if neutral:
            lbs = _neutral_lbs(lbs)

        identity = torch.tensor(data["identity_coeffs"][indices],  dtype=torch.float32, device=self._device)
        lbs_t    = torch.tensor(lbs,                               dtype=torch.float32, device=self._device)
        face_t   = torch.tensor(data["face_expr_coeffs"][indices], dtype=torch.float32, device=self._device)

        verts, _ = self._mhr(
            identity_coeffs=identity,
            model_parameters=lbs_t,
            face_expr_coeffs=face_t,
            apply_correctives=True,
        )
        return verts.cpu().numpy()  # (B, V, 3) cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _flip_y(v: np.ndarray) -> np.ndarray:
    """Negate Y to flip body right-side-up for posed camera-space vertices."""
    out = v.copy()
    out[:, 1] = -out[:, 1]
    return out


def _make_figure(rows, cols, figsize_per_cell=(5, 6)):
    fw = cols * figsize_per_cell[0]
    fh = rows * figsize_per_cell[1]
    return plt.figure(figsize=(fw, fh))


def _add_ax(fig, n_rows, n_cols, row, col, title, show_title):
    ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection="3d")
    if show_title:
        ax.set_title(title, fontsize=9, pad=4)
    return ax


def _solo(ax, v_m, stride, cmap="viridis"):
    """Single-model scatter in metres."""
    _set_equal_axes(ax, v_m)
    _scatter3d(ax, v_m, stride=stride, cmap=cmap)


def _overlay(ax, va_m, vb_m, stride, ca="steelblue", cb="tomato", la="A", lb="B"):
    """Two-model overlay, both in metres."""
    all_v = np.concatenate([va_m, vb_m], axis=0)
    _set_equal_axes(ax, all_v)
    _scatter3d(ax, va_m, color=ca, s=0.5, alpha=0.5, label=la, stride=stride)
    _scatter3d(ax, vb_m, color=cb, s=0.5, alpha=0.5, label=lb, stride=stride)


def _plot_posed_row(fig, n_rows, n_cols, row,
                    smpl_v, mhr_fixed_v, mhr_cond_v, stride):
    """
    Posed columns (SMPL and MHR already flipped before calling):
      SMPL | MHR_fixed | Overlay(SMPL/fixed) [| MHR_cond | Overlay(SMPL/cond) | Overlay(fixed/cond)]
    """
    show_title = (row == 0)
    col = 0

    # SMPL-X
    ax = _add_ax(fig, n_rows, n_cols, row, col, "SMPL-X", show_title); col += 1
    _solo(ax, smpl_v, stride)

    if mhr_fixed_v is not None:
        fv_m = mhr_fixed_v / 100.0
        ax = _add_ax(fig, n_rows, n_cols, row, col, "MHR fixed", show_title); col += 1
        _solo(ax, fv_m, stride)

        ax = _add_ax(fig, n_rows, n_cols, row, col, "Overlay SMPL / MHR fixed", show_title); col += 1
        _overlay(ax, smpl_v, fv_m, stride, la="SMPL-X", lb="MHR fixed")

    if mhr_cond_v is not None:
        cv_m = mhr_cond_v / 100.0
        ax = _add_ax(fig, n_rows, n_cols, row, col, "MHR cond.", show_title); col += 1
        _solo(ax, cv_m, stride)

        ax = _add_ax(fig, n_rows, n_cols, row, col, "Overlay SMPL / MHR cond.", show_title); col += 1
        _overlay(ax, smpl_v, cv_m, stride, la="SMPL-X", lb="MHR cond.")

        if mhr_fixed_v is not None:
            ax = _add_ax(fig, n_rows, n_cols, row, col, "Overlay fixed / cond.", show_title); col += 1
            _overlay(ax, fv_m, cv_m, stride, ca="steelblue", cb="tomato",
                     la="MHR fixed", lb="MHR cond.")


def _plot_neutral_row(fig, n_rows, n_cols, row,
                      mhr_fixed_v, mhr_cond_v, stride):
    """
    Neutral columns — no SMPL (shape-only comparison):
      MHR_fixed [| MHR_cond | Overlay(fixed/cond)]
    """
    show_title = (row == 0)
    col = 0

    if mhr_fixed_v is not None:
        fv_m = mhr_fixed_v / 100.0
        ax = _add_ax(fig, n_rows, n_cols, row, col, "MHR fixed (neutral)", show_title); col += 1
        _solo(ax, fv_m, stride)

    if mhr_cond_v is not None:
        cv_m = mhr_cond_v / 100.0
        ax = _add_ax(fig, n_rows, n_cols, row, col, "MHR cond. (neutral)", show_title); col += 1
        _solo(ax, cv_m, stride)

        if mhr_fixed_v is not None:
            ax = _add_ax(fig, n_rows, n_cols, row, col, "Overlay fixed / cond.", show_title); col += 1
            _overlay(ax, fv_m, cv_m, stride, ca="steelblue", cb="tomato",
                     la="MHR fixed", lb="MHR cond.")


def _n_cols_posed(has_fixed, has_cond):
    cols = 1  # SMPL
    if has_fixed: cols += 2  # MHR fixed + overlay
    if has_cond:  cols += 2  # MHR cond + overlay
    if has_fixed and has_cond: cols += 1  # fixed vs cond overlay
    return cols


def _n_cols_neutral(has_fixed, has_cond):
    cols = 0
    if has_fixed: cols += 1
    if has_cond:  cols += 1
    if has_fixed and has_cond: cols += 1
    return max(cols, 1)


def build_and_save_figure(
    npz_name: str,
    visualizer: GTVisualizer,
    extra_data: dict,
    fixed_data: dict | None,
    cond_data: dict | None,
    frame_indices: np.ndarray,
    output_dir: str,
    neutral: bool,
    stride: int = 8,
):
    tag = "neutral" if neutral else "posed"
    n_frames   = len(frame_indices)
    has_fixed  = fixed_data is not None
    has_cond   = cond_data  is not None
    n_cols     = (_n_cols_neutral(has_fixed, has_cond) if neutral
                  else _n_cols_posed(has_fixed, has_cond))

    fig = _make_figure(n_frames, n_cols)
    fig.suptitle(f"{npz_name}  [{tag}]", fontsize=10, y=1.002)

    smpl_verts  = (None if neutral else
                   visualizer.smplx_vertices(extra_data, frame_indices, neutral=False))
    fixed_verts = (visualizer.mhr_vertices(fixed_data, frame_indices, neutral=neutral)
                   if has_fixed else None)
    cond_verts  = (visualizer.mhr_vertices(cond_data,  frame_indices, neutral=neutral)
                   if has_cond  else None)

    for i in range(n_frames):
        if neutral:
            sv = None
            fv = fixed_verts[i] if fixed_verts is not None else None
            cv = cond_verts[i]  if cond_verts  is not None else None
        else:
            sv = _flip_y(smpl_verts[i])
            fv = _flip_y(fixed_verts[i]) if fixed_verts is not None else None
            cv = _flip_y(cond_verts[i])  if cond_verts  is not None else None

        # Row label
        fig.text(0.002, 1.0 - (i + 0.5) / n_frames,
                 f"fr {frame_indices[i]}  serno {extra_data['serno'][frame_indices[i]]}",
                 ha="left", va="center", fontsize=7, transform=fig.transFigure)

        if neutral:
            _plot_neutral_row(fig, n_frames, n_cols, i, fv, cv, stride)
        else:
            _plot_posed_row(fig, n_frames, n_cols, i, sv, fv, cv, stride)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{os.path.splitext(npz_name)[0]}_{tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(extra_data: dict, n_frames: int) -> np.ndarray:
    """
    Select n_frames evenly spread across unique sernos for variety.
    If there are fewer sernos than n_frames, fall back to evenly spaced indices.
    """
    sernos = extra_data["serno"]
    unique = np.unique(sernos)
    if len(unique) >= n_frames:
        chosen_sernos = unique[np.linspace(0, len(unique) - 1, n_frames, dtype=int)]
        return np.array([np.where(sernos == s)[0][0] for s in chosen_sernos])
    else:
        return np.linspace(0, len(sernos) - 1, n_frames, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise BEDLAM GT quality")
    parser.add_argument("--n_frames",   type=int, default=6,
                        help="Number of frames to visualise per sequence")
    parser.add_argument("--output_dir", type=str, default="vis_gt",
                        help="Directory to save PNG figures")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--stride",     type=int, default=8,
                        help="Vertex subsampling stride for scatter plots")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    viz = GTVisualizer(device)
    print("Models loaded.")

    for npz_name in SUBURB_B_FILES:
        print(f"\n=== {npz_name} ===")

        extra_path = os.path.join(EXTRA_PATH, npz_name)
        if not os.path.exists(extra_path):
            print(f"  [skip] extra npz not found: {extra_path}")
            continue

        extra_data = dict(np.load(extra_path, allow_pickle=False))

        fixed_path = os.path.join(FIXED_PATH, npz_name)
        fixed_data = dict(np.load(fixed_path, allow_pickle=False)) if os.path.exists(fixed_path) else None
        if fixed_data is None:
            print("  [warn] fixed npz not found — MHR (fixed) panels will be empty")

        cond_path  = os.path.join(CONDITIONED_PATH, npz_name)
        cond_data  = dict(np.load(cond_path, allow_pickle=False)) if os.path.exists(cond_path) else None
        if cond_data is None:
            print("  [info] conditioned npz not yet available — will be skipped")

        frame_indices = select_frames(extra_data, args.n_frames)
        print(f"  frames: {frame_indices.tolist()}  "
              f"sernos: {extra_data['serno'][frame_indices].tolist()}")

        for neutral in (False, True):
            build_and_save_figure(
                npz_name     = npz_name,
                visualizer   = viz,
                extra_data   = extra_data,
                fixed_data   = fixed_data,
                cond_data    = cond_data,
                frame_indices = frame_indices,
                output_dir   = args.output_dir,
                neutral      = neutral,
                stride       = args.stride,
            )


if __name__ == "__main__":
    main()
