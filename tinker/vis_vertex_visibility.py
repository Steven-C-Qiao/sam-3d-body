"""Visualise per-vertex visibility of a single MHR mesh in 3D next to the image.

Loads a single BEDLAM sample, runs the MHR body model to get mesh vertices,
loads the precomputed `_vertex_visibility.npz` mask for that sample, and
renders a two-panel figure:
  - left:  3D scatter of the 18,439 vertices, visible=green / occluded=red
  - right: the corresponding RGB image with the GT bbox overlaid

Usage:
    python tinker/vis_vertex_visibility.py \
        --dataset 20221010_3_1000_batch01hand_6fps \
        --index 0 \
        --out /tmp/vertex_vis.png
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


BEDLAM_ROOT = "/scratch/cq244/BEDLAM/data"
NPZ_DIR = os.path.join(BEDLAM_ROOT, "training_labels/all_npz_12_training_mhr_conditioned")
VIS_DIR = os.path.join(BEDLAM_ROOT, "training_labels/visibility_labels")
IMG_ROOT = os.path.join(BEDLAM_ROOT, "training_images")
MHR_MODEL_PATH = "/scratch/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"


def align(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x = x - c
    x[..., [1, 2]] *= -1
    x = x + c
    x[..., [1, 2]] *= -1
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="20221010_3_1000_batch01hand_6fps")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=0,
                        help="Optional subsample cap on vertices (0 = all 18,439)")
    parser.add_argument("--y_up", action=argparse.BooleanOptionalAction, default=True,
                        help="Flip Y for plotting so the body appears head-up "
                             "(MHR+align leaves Y pointing down, camera convention).")
    parser.add_argument("--out", default="/tmp/vertex_vis.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = args.index

    # Labels
    npz_path = os.path.join(NPZ_DIR, f"{args.dataset}.npz")
    data = np.load(npz_path, allow_pickle=True)

    vv_path = os.path.join(VIS_DIR, f"{args.dataset}_vertex_visibility.npz")
    vertex_vis = np.load(vv_path)["vertex_visibility"][idx].astype(bool)  # (18439,)

    # Image
    img_path = os.path.join(IMG_ROOT, args.dataset, "png", data["imgname"][idx])
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_ROOT, args.dataset, data["imgname"][idx])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if "closeup" in args.dataset.lower():
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    H, W = img.shape[:2]

    # MHR forward (match trainer.preprocess: save translation, zero it before forward)
    mhr_model = torch.jit.load(MHR_MODEL_PATH, map_location=device)
    ident = torch.tensor(data["identity_coeffs"][idx:idx + 1], dtype=torch.float32, device=device)
    lbs = torch.tensor(data["lbs_model_params"][idx:idx + 1], dtype=torch.float32, device=device)
    fexpr = torch.tensor(data["face_expr_coeffs"][idx:idx + 1], dtype=torch.float32, device=device)
    lbs[:, :3] = 0

    c = torch.tensor([0., 0.923986, 0.], device=device)
    with torch.no_grad():
        verts, _ = mhr_model(ident, lbs, fexpr)
        verts = verts / 100.0
        verts = align(verts, c)
    verts_np = verts[0].cpu().numpy()  # (V, 3)
    assert verts_np.shape[0] == vertex_vis.shape[0], (verts_np.shape, vertex_vis.shape)

    # Optional subsample for faster plotting while keeping the split reps.
    if args.max_points and args.max_points < verts_np.shape[0]:
        rng = np.random.default_rng(0)
        idx_sub = rng.choice(verts_np.shape[0], size=args.max_points, replace=False)
        verts_np = verts_np[idx_sub]
        vertex_vis = vertex_vis[idx_sub]

    vis_mean = float(vertex_vis.mean())
    plot_verts = verts_np.copy()
    if args.y_up:
        plot_verts[:, 1] *= -1  # flip Y so head points up in the 3D plot
    vis_xyz = plot_verts[vertex_vis]
    occ_xyz = plot_verts[~vertex_vis]

    # ---------- figure ----------
    fig = plt.figure(figsize=(16, 8))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(vis_xyz[:, 0], vis_xyz[:, 1], vis_xyz[:, 2],
                 c="lime", s=1.5, alpha=0.6, label=f"visible ({vis_xyz.shape[0]})")
    ax3d.scatter(occ_xyz[:, 0], occ_xyz[:, 1], occ_xyz[:, 2],
                 c="red", s=1.5, alpha=0.35, label=f"occluded ({occ_xyz.shape[0]})")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("-y (up)" if args.y_up else "y")
    ax3d.set_zlabel("z")
    ax3d.set_title(f"vertex visibility  (mean visible = {vis_mean:.3f})")
    ax3d.legend(loc="upper right", fontsize=9)
    # Equal aspect ratio
    mins = plot_verts.min(axis=0); maxs = plot_verts.max(axis=0)
    ctr = 0.5 * (mins + maxs); rng = 0.5 * (maxs - mins).max()
    ax3d.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax3d.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax3d.set_zlim(ctr[2] - rng, ctr[2] + rng)

    cx, cy = data["center"][idx]
    bs = float(data["scale"][idx]) * 200.0 / 2.0

    ax_img = fig.add_subplot(1, 2, 2)
    ax_img.imshow(img)
    ax_img.add_patch(plt.Rectangle((cx - bs, cy - bs), 2 * bs, 2 * bs,
                                   fill=False, edgecolor="yellow", linewidth=1.2))
    ax_img.set_xlim(0, W); ax_img.set_ylim(H, 0)
    ax_img.set_title(f"{args.dataset} / {data['imgname'][idx]}")
    ax_img.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}  (visible={vis_xyz.shape[0]}/{verts_np.shape[0]}, mean={vis_mean:.3f})")


if __name__ == "__main__":
    main()
