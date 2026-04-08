#!/usr/bin/env python
"""
Visualise why GT and predicted meshes project correctly onto the image
despite using different coordinate conventions.

Shows the 3D geometry *before* and *after* the alignment fix in a 2x2 grid:
  Top-left:     GT path (as passed to renderer)
  Top-right:    Pred path (as passed to renderer)
  Bottom-left:  Aligned -- convert pred to GT convention
  Bottom-right: Reprojection check on original image

Usage:
    python scripts/visualise_projection_alignment.py -L <checkpoint_path>
    python scripts/visualise_projection_alignment.py   # auto-finds latest checkpoint
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
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(exp_root="exp"):
    """Glob for *.ckpt under exp/ and return the most recently modified one."""
    pattern = os.path.join(exp_root, "**", "*.ckpt")
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {exp_root}/")
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def subsample_verts(verts, max_pts=2000):
    """Uniformly subsample vertices for scatter plotting."""
    n = verts.shape[0]
    if n <= max_pts:
        return verts
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    return verts[idx]


def set_equal_aspect_3d(ax, points):
    """Set equal axis limits for a 3D plot centred on the point cloud."""
    mid = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0 * 1.2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def project_to_2d(verts_3d, cam_t, K):
    """
    Project 3D points to 2D image coordinates.
    verts_3d: (N, 3)  -- already in the camera coordinate system
    cam_t:    (3,)     -- camera translation added to points
    K:        (3, 3)   -- intrinsic matrix
    Returns: (N, 2) pixel coords
    """
    pts = verts_3d + cam_t[None, :]  # (N, 3)
    z = pts[:, 2:3].clip(min=1e-6)
    pts_norm = pts / z  # (N, 3)
    px = (K @ pts_norm.T).T  # (N, 3)
    return px[:, :2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise projection alignment")
    parser.add_argument(
        "--load_from_ckpt", "-L", type=str, default=None,
        help="Path to checkpoint. If not given, auto-finds latest under exp/.",
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="GPU index (single GPU only).",
    )
    parser.add_argument(
        "--batch_idx", type=int, default=0,
        help="Which sample in the batch to visualise.",
    )
    parser.add_argument(
        "--config", "-C", type=str, default=None,
        help="Optional YAML config override.",
    )
    args = parser.parse_args()

    # ---- Environment setup ----
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Config ----
    cfg = get_config_defaults()
    ckpt_path = args.load_from_ckpt
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(
            os.path.join(os.path.dirname(__file__), "..", "exp")
        )
        print(f"[Auto-found checkpoint] {ckpt_path}")
    else:
        print(f"[Using checkpoint] {ckpt_path}")

    # Try to load config from the experiment directory of the checkpoint
    exp_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_yaml = os.path.join(exp_dir, "config.yaml")
    if os.path.isfile(config_yaml):
        print(f"[Loading config from] {config_yaml}")
        cfg.merge_from_file(config_yaml)
    if args.config is not None:
        cfg.merge_from_file(args.config)

    # Use small batch for visualisation
    cfg.DATASET.BATCH_SIZE = 2
    cfg.DATASET.NUM_WORKERS = 2

    # ---- Build model ----
    print("[Building model...]")
    model = Trainer(cfg=cfg, vis_save_dir="/tmp/vis_alignment")

    # Load checkpoint weights into model.model
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state_dict[key[6:]] = value
        elif not any(k in key for k in ["optimizer", "lr_scheduler", "epoch", "global_step", "callbacks"]):
            model_state_dict[key] = value
    if model_state_dict:
        missing, unexpected = model.model.load_state_dict(model_state_dict, strict=False)
        print(f"[Loaded {len(model_state_dict)} params] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[WARNING] No model parameters found in checkpoint!")

    model = model.to(device)
    model.eval()

    # ---- Build val dataset (first BEDLAM val set) ----
    val_ds_names = cfg.DATASET.VAL_DS.split("_")
    print(f"[Val datasets] {val_ds_names}")
    ds = BEDLAMDataset(options=cfg.DATASET, dataset=val_ds_names[0])
    loader = DataLoader(
        ds, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS, collate_fn=bedlam_collate,
    )

    # ---- Get one batch ----
    print("[Loading batch...]")
    batch = next(iter(loader))

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # ---- Preprocess (computes gt_verts_w_transl etc.) ----
    print("[Preprocessing...]")
    batch = model.preprocess(batch)

    # ---- Forward pass ----
    print("[Running forward pass...]")
    try:
        with torch.no_grad():
            outputs = model.model(batch, num_samples=1)
    except Exception as e:
        print(f"[ERROR in forward pass] {e}")
        import traceback
        traceback.print_exc()
        print("[Attempting forward with num_samples=0...]")
        with torch.no_grad():
            outputs = model.model(batch, num_samples=0)

    # ---- Extract tensors for sample `bi` ----
    bi = args.batch_idx
    print(f"\n{'='*60}")
    print(f"Extracting data for batch index {bi}")
    print(f"{'='*60}")

    # GT verts: raw MHR output (no Y/Z flip), body translation baked in
    # Shape: batch["gt_verts_w_transl"] is (B, N_verts, 3)
    gt_verts = batch["gt_verts_w_transl"][bi].cpu().numpy()
    print(f"gt_verts shape: {gt_verts.shape}")

    # GT camera translation: extrinsic camera translation (last column of cam_ext)
    # cam_ext is (B, 4, 4) or (B, 3, 4) -- take first 3 rows, last column
    gt_cam_t = batch["cam_ext"][bi, :3, 3].cpu().numpy()
    print(f"gt_cam_t: {gt_cam_t}")

    # Body translation: first 3 elements of model_params, scaled from MHR units to metres
    body_transl = batch["model_params"][bi, :3].cpu().numpy() / 100.0
    print(f"body_transl (metres): {body_transl}")

    # Camera intrinsics
    K = batch["cam_int"][bi].cpu().numpy()
    print(f"K:\n{K}")

    # Predicted verts (mean prediction) -- already Y/Z flipped in sam3d_body.py:869
    pred_verts = outputs["mhr"]["pred_vertices"][bi].cpu().numpy()
    print(f"pred_verts shape: {pred_verts.shape}")

    # Predicted camera translation
    pred_cam_t = outputs["mhr"]["pred_cam_t"][bi].cpu().numpy()
    print(f"pred_cam_t: {pred_cam_t}")

    # Root joint positions (joint index 1 in MHR ordering, matching vis_utils.py:323)
    # GT: gt_joint_coords is already /100 (metres), in same frame as gt_verts (no Y/Z flip)
    gt_root_joint = batch["gt_joint_coords"][bi, 1, :].cpu().numpy()  # (3,)
    print(f"gt_root_joint: {gt_root_joint}")
    # Pred: pred_joint_coords is (B, N_joints, 3), Y/Z flipped, no translation (mean prediction)
    pred_root_joint = outputs["mhr"]["pred_joint_coords"][bi, 1, :].cpu().numpy()  # (3,)
    print(f"pred_root_joint: {pred_root_joint}")

    # Try to get the original image
    imgname = batch.get("imgname", None)
    img_ori = None
    if imgname is not None:
        if isinstance(imgname, (list, tuple)):
            imgpath = imgname[bi]
        else:
            imgpath = imgname
        print(f"imgname: {imgpath}")
        if os.path.isfile(imgpath):
            img_ori = cv2.imread(imgpath)
            if img_ori is not None:
                img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                print(f"Loaded image: {img_ori.shape}")
    # Fallback: use img_ori from batch (list of numpy arrays from collate)
    if img_ori is None and "img_ori" in batch:
        img_ori_list = batch["img_ori"]
        if isinstance(img_ori_list, list) and len(img_ori_list) > bi:
            img_ori = img_ori_list[bi]
            if img_ori is not None:
                if img_ori.shape[-1] == 3 and img_ori.dtype == np.uint8:
                    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                print(f"Using img_ori from batch: {img_ori.shape}")

    # ---- Compute derived quantities ----

    # Renderer applies 180-deg rotation around X to the mesh: [x, y, z] -> [x, -y, -z]
    # Renderer also negates cam_t[0] before placing camera.

    # GT path as passed to renderer (ORIGINAL convention):
    #   verts = gt_verts (body_transl baked in, no Y/Z flip applied by code)
    #   cam_t = gt_cam_t = cam_ext[:3, 3]

    x = -(
        (gt_root_joint * np.array([1, -1, -1])) - (np.array([0., 0.92398697, 0.]))
        * np.array([-1, 1, 1])
    )
    y = gt_root_joint - np.array([0., 0.92398697, 0.]) * np.array([1, -1, -1])

    gt_cam_t += x 
    gt_verts -= y[None, :]
    
    gt_verts_rendered = gt_verts * np.array([1, -1, -1])  # renderer's 180-X rotation
    gt_cam_rendered = gt_cam_t.copy()
    gt_cam_rendered[0] *= -1  # renderer negates X

    # Pred path as passed to renderer:
    #   verts = pred_verts (already Y/Z flipped by sam3d_body.py:869), no translation
    #   renderer applies 180-X rotation: [x, -y, -z] -> [x, y, z] -- double flip on Y,Z
    pred_verts_rendered = pred_verts * np.array([1, -1, -1])  # renderer's 180-X rotation
    pred_cam_rendered = pred_cam_t.copy()
    pred_cam_rendered[0] *= -1  # renderer negates X

    # Root joints in renderer space (apply same 180-X flip as verts)
    gt_root_rendered = gt_root_joint * np.array([1, -1, -1])
    pred_root_rendered = pred_root_joint * np.array([1, -1, -1])

    ### gt_verts = gt_verts - ((np.array([0., 0.92398697, 0.]) - gt_root_rendered)[None, :])
    ### gt_cam_t = gt_cam_t + ((np.array([0., 0.92398697, 0.]) - gt_root_rendered))

    # gt_verts_rendered = gt_verts_rendered + ((np.array([0., 0.92398697, 0.]) - gt_root_rendered)[None, :])
    # gt_cam_rendered = gt_cam_rendered + ((np.array([0., 0.92398697, 0.]) - gt_root_rendered))

    # ---- Create figure ----
    fig = plt.figure(figsize=(18, 9))

    # ---------- Left: GT path (as passed to renderer) ----------
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(gt_verts_rendered[:, 0], gt_verts_rendered[:, 1], gt_verts_rendered[:, 2],
                c="steelblue", s=1, alpha=0.5, label="GT verts (after renderer 180-X)")
    ax1.scatter(*gt_cam_rendered, c="red", s=100, marker="^", label="GT camera (X-negated)")
    ax1.text(*gt_cam_rendered, "  GT cam", fontsize=7, color="red")
    # Draw arrow from camera to body centre
    body_centre = gt_verts_rendered.mean(axis=0)
    ax1.plot([gt_cam_rendered[0], body_centre[0]],
             [gt_cam_rendered[1], body_centre[1]],
             [gt_cam_rendered[2], body_centre[2]], "r--", alpha=0.5)
    # Overlay pred path on the same axes
    ax1.scatter(pred_verts_rendered[:, 0], pred_verts_rendered[:, 1], pred_verts_rendered[:, 2],
                c="darkorange", s=1, alpha=0.5, label="Pred verts (Y/Z flip + renderer 180-X)")
    ax1.scatter(*pred_cam_rendered, c="darkred", s=100, marker="s", label="Pred camera (X-negated)")
    ax1.text(*pred_cam_rendered, "  Pred cam", fontsize=7, color="darkred")
    body_centre_pred = pred_verts_rendered.mean(axis=0)
    ax1.plot([pred_cam_rendered[0], body_centre_pred[0]],
             [pred_cam_rendered[1], body_centre_pred[1]],
             [pred_cam_rendered[2], body_centre_pred[2]], "m--", alpha=0.5)
    ax1.set_title(
        "GT (blue) + Pred (orange) as passed to renderer\n(after renderer 180-X rotation + X-negated cam)",
        fontsize=10,
    )
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(fontsize=7, loc="upper left")



    print(f"\n[Subplot 1 — renderer space]")
    print(f"  GT   camera : {gt_cam_rendered}")
    print(f"  Pred camera : {pred_cam_rendered}")
    print(f"  GT   root   : {gt_root_rendered}")
    print(f"  Pred root   : {pred_root_rendered}")

    # all_top = np.vstack([gt_verts_rendered, gt_cam_rendered[None, :], pred_verts_rendered, pred_cam_rendered[None, :]])
    all_top = np.vstack([gt_verts_rendered, pred_verts_rendered])
    set_equal_aspect_3d(ax1, all_top)

    # ---------- Right: Reprojection check ----------
    ax4 = fig.add_subplot(1, 2, 2)

    if img_ori is not None:
        ax4.imshow(img_ori)
    else:
        ax4.text(0.5, 0.5, "No image available", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=14)

    # GT reprojection — two versions should overlap perfectly:
    #   Original:  gt_verts (with transl) + gt_cam_t
    #   Aligned:   gt_verts_no_transl    + gt_cam_t_aligned  (transl shifted to cam_t)
    gt_proj_sub = subsample_verts(gt_verts, max_pts=500)



    gt_2d_original = project_to_2d(gt_proj_sub, gt_cam_t, K)

    # gt_proj_no_transl_sub = subsample_verts(gt_verts_no_transl, max_pts=500)
    # gt_2d_aligned = project_to_2d(gt_proj_no_transl_sub, gt_cam_t_aligned, K)

    # Pred reprojection (pred_verts Y/Z flipped + pred_cam_t)
    pred_proj_sub = subsample_verts(pred_verts, max_pts=500)
    pred_2d = project_to_2d(pred_proj_sub, pred_cam_t, K)

    ax4.scatter(gt_2d_original[:, 0], gt_2d_original[:, 1],
                c="steelblue", s=6, alpha=0.5, label="GT original reproj")
    # ax4.scatter(gt_2d_aligned[:, 0], gt_2d_aligned[:, 1],
    #             c="cyan", s=2, alpha=0.6, label="GT aligned reproj (should overlap blue)")
    ax4.scatter(pred_2d[:, 0], pred_2d[:, 1], c="darkorange", s=2, alpha=0.4, label="Pred reproj")
    ax4.set_title("Reprojection check\nGT (blue) vs Pred (orange) on original image", fontsize=10)
    ax4.legend(fontsize=8, loc="upper right")
    if img_ori is not None:
        ax4.set_xlim(0, img_ori.shape[1])
        ax4.set_ylim(img_ori.shape[0], 0)
    ax4.set_aspect("equal")

    # ---- Annotations with numeric info ----
    # reproj_err = np.mean(np.linalg.norm(gt_2d_original - gt_2d_aligned, axis=1))
    info_lines = [
        f"gt_cam_t          = [{gt_cam_t[0]:.3f}, {gt_cam_t[1]:.3f}, {gt_cam_t[2]:.3f}]",
        f"body_transl       = [{body_transl[0]:.4f}, {body_transl[1]:.4f}, {body_transl[2]:.4f}]",
        f"pred_cam_t        = [{pred_cam_t[0]:.3f}, {pred_cam_t[1]:.3f}, {pred_cam_t[2]:.3f}]",
        # f"reproj error (original vs aligned GT, px): {reproj_err:.6f}  <-- should be ~0",
    ]
    fig.text(0.02, 0.02, "\n".join(info_lines), fontsize=8, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    out_path = os.path.join(os.path.dirname(__file__), "debug_alignment.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved figure to] {out_path}")

    # Also try to show (will be no-op on headless servers)
    try:
        plt.show()
    except Exception:
        pass

    print("[Done]")


if __name__ == "__main__":
    main()
