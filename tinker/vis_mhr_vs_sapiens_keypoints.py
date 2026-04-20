"""Visualise the 127 MHR skeleton keypoints vs the 308 Sapiens keypoints.

Loads a single BEDLAM sample, runs the MHR body model to get mesh vertices +
127 skeleton joints, applies the learned `keypoint_mapping` regressor to
produce the 308 Sapiens-style keypoints, projects both sets into the image,
and saves a side-by-side figure.

Usage:
    python scripts/vis_mhr_vs_sapiens_keypoints.py \
        --dataset 20221010_3_1000_batch01hand_6fps \
        --index 0 \
        --out /tmp/kps_127_vs_308.png
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


BEDLAM_ROOT = "/scratch/cq244/BEDLAM/data"
NPZ_DIR = os.path.join(BEDLAM_ROOT, "training_labels/all_npz_12_training_mhr_conditioned")
IMG_ROOT = os.path.join(BEDLAM_ROOT, "training_images")
MHR_MODEL_PATH = "/scratch/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
CKPT_PATH = "/scratch/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"


def project(points_3d: torch.Tensor, trans_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Project (B,N,3) 3D points (in body/model frame) into (B,N,2) pixel coords."""
    p = points_3d + trans_cam.unsqueeze(1)               # (B, N, 3)
    p = p / p[..., -1:].clamp(min=1e-6)                  # perspective divide
    p = torch.einsum("bij,bnj->bni", K, p)               # apply intrinsics
    return p[..., :2]


def load_visibility(npz_path: str, index: int):
    vb_path = npz_path.replace(
        "all_npz_12_training_mhr_conditioned", "visibility_labels"
    )[:-4] + "_visibility_308.npz"
    if not os.path.exists(vb_path):
        return None, None
    d = np.load(vb_path)
    return d["visibility"][index].astype(bool), d["visibility_308"][index].astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="20221010_3_1000_batch01hand_6fps")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="/tmp/kps_127_vs_308.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    npz_path = os.path.join(NPZ_DIR, f"{args.dataset}.npz")
    data = np.load(npz_path, allow_pickle=True)
    idx = args.index

    img_path = os.path.join(IMG_ROOT, args.dataset, "png", data["imgname"][idx])
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_ROOT, args.dataset, data["imgname"][idx])
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if "closeup" in args.dataset.lower():
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    H, W = img.shape[:2]
    print(f"image {img_path} shape={img.shape}")

    # MHR forward: identity (shape), lbs params (pose+trans+scale), face expr
    mhr_model = torch.jit.load(MHR_MODEL_PATH, map_location=device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    keypoint_mapping = ckpt["head_pose.keypoint_mapping"].to(device)   # (308, V+J)

    ident = torch.tensor(data["identity_coeffs"][idx:idx + 1], dtype=torch.float32, device=device)
    lbs = torch.tensor(data["lbs_model_params"][idx:idx + 1], dtype=torch.float32, device=device)
    fexpr = torch.tensor(data["face_expr_coeffs"][idx:idx + 1], dtype=torch.float32, device=device)
    # Match trainer.preprocess: save translation, zero it before MHR forward.
    mhr_transl = lbs[:, :3].clone() / 10.0
    lbs[:, :3] = 0

    def align(x, c):
        x = x - c
        x[..., [1, 2]] *= -1
        x = x + c
        x[..., [1, 2]] *= -1
        return x

    c = torch.tensor([0., 0.923986, 0.], device=device)

    with torch.no_grad():
        verts, skeleton = mhr_model(ident, lbs, fexpr)
        verts = verts / 100.0
        joints127 = skeleton[:, :, :3] / 100.0
        verts = align(verts, c)
        joints127 = align(joints127, c)
        vj = torch.cat([verts, joints127], dim=1)
        kp308 = (
            keypoint_mapping @ vj.permute(1, 0, 2).flatten(1, 2)
        ).reshape(-1, vj.shape[0], 3).permute(1, 0, 2)     # (1, 308, 3)

    K = torch.tensor(data["cam_int"][idx:idx + 1], dtype=torch.float32, device=device)
    cam_ext = torch.tensor(data["cam_ext"][idx:idx + 1], dtype=torch.float32, device=device)
    trans_cam_npz = torch.tensor(data["trans_cam"][idx:idx + 1], dtype=torch.float32, device=device)
    # bedlam_dataset.py:227 folds trans_cam into cam_ext[:3,3] before trainer sees it;
    # trainer then does trans_cam = cam_ext[:3,3] and adds 2c + mhr_transl.
    trans_cam = cam_ext[:, :3, 3] + trans_cam_npz + 2 * c + mhr_transl

    kp127_2d = project(joints127, trans_cam, K)[0].cpu().numpy()
    kp308_2d = project(kp308, trans_cam, K)[0].cpu().numpy()

    vis127, vis308 = load_visibility(npz_path, idx)

    cx, cy = data["center"][idx]
    bs = float(data["scale"][idx]) * 200.0 / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(img)
    if vis127 is not None:
        axes[0].scatter(kp127_2d[vis127, 0], kp127_2d[vis127, 1],
                        s=22, c="lime", edgecolor="black", linewidths=0.4, label="visible")
        axes[0].scatter(kp127_2d[~vis127, 0], kp127_2d[~vis127, 1],
                        s=22, c="red", edgecolor="black", linewidths=0.4, label="occluded")
        axes[0].legend(loc="lower right")
    else:
        axes[0].scatter(kp127_2d[:, 0], kp127_2d[:, 1], s=22, c="lime", edgecolor="black", linewidths=0.4)
    axes[0].set_title(f"127 MHR skeleton joints (idx={idx})")
    axes[0].add_patch(plt.Rectangle((cx-bs, cy-bs), 2*bs, 2*bs, fill=False, edgecolor="yellow", linewidth=1.2))
    axes[0].set_xlim(0, W); axes[0].set_ylim(H, 0)
    axes[0].axis("off")
    print("kp127 x range:", kp127_2d[:, 0].min(), kp127_2d[:, 0].max(),
          "y range:", kp127_2d[:, 1].min(), kp127_2d[:, 1].max())
    print("kp308 x range:", kp308_2d[:, 0].min(), kp308_2d[:, 0].max(),
          "y range:", kp308_2d[:, 1].min(), kp308_2d[:, 1].max())

    axes[1].imshow(img)
    if vis308 is not None:
        axes[1].scatter(kp308_2d[vis308, 0], kp308_2d[vis308, 1],
                        s=6, c="cyan", edgecolor="black", linewidths=0.2, label="visible")
        axes[1].scatter(kp308_2d[~vis308, 0], kp308_2d[~vis308, 1],
                        s=6, c="red", edgecolor="black", linewidths=0.2, label="occluded")
        # Highlight the first 70 (the ones actually used in training).
        axes[1].scatter(kp308_2d[:70, 0], kp308_2d[:70, 1],
                        s=40, facecolor="none", edgecolor="yellow", linewidths=0.8,
                        label="first 70 (used in loss)")
        axes[1].legend(loc="lower right")
    else:
        axes[1].scatter(kp308_2d[:, 0], kp308_2d[:, 1], s=6, c="cyan",
                        edgecolor="black", linewidths=0.2)
    axes[1].set_title("308 Sapiens keypoints (via keypoint_mapping)")
    axes[1].set_xlim(0, W); axes[1].set_ylim(H, 0)
    axes[1].axis("off")

    fig.suptitle(f"{args.dataset} / {data['imgname'][idx]}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
