#!/usr/bin/env python
"""
Visualise vertex outputs from Trainer.preprocess().

Loads one batch, runs preprocess(), and produces a figure showing:
  1. GT vertices (body frame, as stored in batch) — 3D scatter
  2. GT vertices after Y/Z flip (as used by loss) — 3D scatter
  3. GT vertices projected onto the original image — 2D overlay
  4. GT vertices rendered via the Renderer — mesh overlay on image

Usage:
    python scripts/visualise_preprocess_vertices.py -L <checkpoint_path>
    python scripts/visualise_preprocess_vertices.py  # auto-finds latest checkpoint
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam_3d_body.configs.config import get_config_defaults
from sam_3d_body.trainer import Trainer
from sam_3d_body.data.bedlam_dataset import DatasetHMR as BEDLAMDataset, bedlam_collate
from sam_3d_body.visualization.renderer import Renderer
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(exp_root="exp"):
    pattern = os.path.join(exp_root, "**", "*.ckpt")
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {exp_root}/")
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def subsample_verts(verts, max_pts=2000):
    n = verts.shape[0]
    if n <= max_pts:
        return verts
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    return verts[idx]


def set_equal_aspect_3d(ax, points):
    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0 * 1.2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def project_to_2d(verts_3d, cam_t, K):
    """Project 3D points to pixel coordinates: verts_3d (N,3), cam_t (3,), K (3,3)."""
    pts = verts_3d + cam_t[None, :]
    z = pts[:, 2:3].clip(min=1e-6)
    pts_norm = pts / z
    px = (K @ pts_norm.T).T
    return px[:, :2]


def xyz_label(name, pt):
    return f"{name} ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})"


def scatter_3d(ax, verts, title, color="steelblue"):
    v = subsample_verts(verts, max_pts=1500)
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.3, alpha=0.5, c=color)
    set_equal_aspect_3d(ax, v)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--ckpt", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default="preprocess_vertices.png")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index within batch")
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")

    cfg = get_config_defaults()
    # trainer = Trainer.load_from_checkpoint(ckpt_path, cfg=cfg, strict=False)
    trainer = Trainer(cfg=cfg)
    trainer.eval()
    trainer.cuda()

    # Load one batch
    ds = BEDLAMDataset(cfg.DATASET, cfg.DATASET.DATASETS_AND_RATIOS.split("_")[0])
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=bedlam_collate, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        batch = trainer.preprocess(batch)
        outputs = trainer(batch, num_samples=0)

    i = args.sample_idx

    # --- Extract data ---
    # batch["vertices"]: GT verts from new_preprocess (Y/Z-flipped, aligned with predictions)
    gt_verts = batch["vertices"][i].cpu().numpy()               # (V, 3) Y/Z-flipped

    # Predicted vertices (Y/Z-flipped by MHR head) + predicted camera translation
    pred_verts = outputs["mhr"]["pred_vertices"][i].cpu().numpy()          # (V, 3) Y/Z-flipped
    pred_cam_t = outputs["mhr"]["pred_cam_t"][i].cpu().numpy()             # (3,)
    pred_joint_coords = outputs["mhr"]["pred_joint_coords"][i].cpu().numpy()  # (J, 3) Y/Z-flipped
    pred_root = pred_joint_coords[0]    # (3,)
    pred_pelvis = pred_joint_coords[1]  # (3,)

    root_joint = batch["joint_coords"][i, 0].cpu().numpy()  # (3,) joint 0 = root
    pelvis_joint = batch["joint_coords"][i, 1].cpu().numpy() # (3,) joint 1 = pelvis

    cam_t = batch["cam_ext"][i, :3, 3].cpu().numpy()            # (3,)
    K = batch["cam_int"][i].cpu().numpy()                       # (3, 3)

    print(cam_t)
    print(pred_cam_t)

    # Original image
    img_ori = batch["img_ori"][i]
    if isinstance(img_ori, torch.Tensor):
        img_ori = img_ori.cpu().numpy()
    img_rgb = img_ori.copy()
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0

    # --- Figure: 2x2 ---
    fig = plt.figure(figsize=(16, 14))

    # Panel 1: GT vertices (Y/Z-flipped, from new_preprocess)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    scatter_3d(ax1, gt_verts, "GT vertices (flipped frame)")
    ax1.scatter(*root_joint, s=80, c="red", marker="x", zorder=10)
    ax1.text(*root_joint, "  " + xyz_label("root", root_joint), fontsize=7, color="red")
    ax1.scatter(*pelvis_joint, s=80, c="magenta", marker="o", zorder=10)
    ax1.text(*pelvis_joint, "  " + xyz_label("pelvis", pelvis_joint), fontsize=7, color="magenta")

    # Panel 2: Overlay GT and predicted vertices
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    scatter_3d(ax2, gt_verts, "GT (blue) vs Pred (green)", color="darkorange")
    scatter_3d(ax2, pred_verts, "", color="limegreen")
    # Re-compute axis limits to fit both sets
    all_verts = np.concatenate([subsample_verts(gt_verts, 1500),
                                subsample_verts(pred_verts, 1500)], axis=0)
    set_equal_aspect_3d(ax2, all_verts)
    # GT joints
    ax2.scatter(*root_joint, s=80, c="red", marker="x", zorder=10)
    ax2.text(*root_joint, "  " + xyz_label("GT root", root_joint), fontsize=7, color="red")
    ax2.scatter(*pelvis_joint, s=80, c="magenta", marker="o", zorder=10)
    ax2.text(*pelvis_joint, "  " + xyz_label("GT pelvis", pelvis_joint), fontsize=7, color="magenta")
    # Predicted joints
    ax2.scatter(*pred_root, s=80, c="darkred", marker="x", zorder=10)
    ax2.text(*pred_root, "  " + xyz_label("pred root", pred_root), fontsize=7, color="darkred")
    ax2.scatter(*pred_pelvis, s=80, c="purple", marker="o", zorder=10)
    ax2.text(*pred_pelvis, "  " + xyz_label("pred pelvis", pred_pelvis), fontsize=7, color="purple")

    # Panel 3: Project GT vertices onto image (flipped verts + adjusted cam_ext)
    ax3 = fig.add_subplot(2, 2, 3)
    px = project_to_2d(gt_verts, cam_t, K)
    ax3.imshow(img_rgb)
    v_sub = subsample_verts(px, max_pts=1500)
    ax3.scatter(v_sub[:, 0], v_sub[:, 1], s=0.3, alpha=0.4, c="lime")
    root_px = project_to_2d(root_joint[None], cam_t, K)[0]
    ax3.scatter(root_px[0], root_px[1], s=80, c="red", marker="x", zorder=10)
    ax3.annotate(xyz_label("root", root_joint), (root_px[0], root_px[1]),
                 fontsize=7, color="red", xytext=(5, -10), textcoords="offset points")
    pelvis_px = project_to_2d(pelvis_joint[None], cam_t, K)[0]
    ax3.scatter(pelvis_px[0], pelvis_px[1], s=80, c="magenta", marker="o", zorder=10)
    ax3.annotate(xyz_label("pelvis", pelvis_joint), (pelvis_px[0], pelvis_px[1]),
                 fontsize=7, color="magenta", xytext=(5, 10), textcoords="offset points")
    ax3.set_title("GT projected onto image (flipped + adjusted cam_ext)", fontsize=10)
    ax3.axis("off")

    # Panel 4: Rendered mesh overlay via Renderer
    # Renderer applies 180-deg X rotation internally, so un-flip GT verts before passing.
    ax4 = fig.add_subplot(2, 2, 4)
    faces = trainer.faces
    focal_length = K[0, 0]
    renderer = Renderer(focal_length=focal_length, faces=faces)
    img_render = (img_rgb * 255.0).astype(np.float32)
    cx, cy = K[0, 2], K[1, 2]
    gt_verts_unflipped = gt_verts.copy()
    gt_verts_unflipped[..., [1, 2]] *= -1  # un-flip for renderer
    rendered = renderer(
        gt_verts_unflipped,
        cam_t,
        img_render,
        camera_center=[cx, cy],
    )
    ax4.imshow(rendered)
    ax4.set_title("Renderer: GT mesh on image", fontsize=10)
    ax4.axis("off")

    fig.suptitle("Preprocess vertex outputs", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
