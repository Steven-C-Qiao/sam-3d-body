"""Visualise body-shape effect of the top uncertainty / disagreement dims.

Renders neutral MHR meshes for ±k·σ_prior perturbations of the shape PCs
identified by `tinker/diag_effective_dim.py`:
  - Top uncertainty: d43, d24, d7, d31, d28
  - Top disagreement: d10, d34, d25, d18, d11

Saves a 10×5 grid (rows=dims, cols=σ values) showing front view, plus
optional 3/4 side view as a separate figure.

Usage:
    python tinker/viz_top_dims.py \
        -E exp/exp_071_crop_shape \
        -L exp/exp_071_crop_shape/saved_models/last.ckpt
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger

sys.path.append(".")


UNC_DIMS = [43, 24, 7, 31, 28]
DIS_DIMS = [10, 34, 25, 18, 11]
SIGMAS = [-3.0, -1.5, 0.0, 1.5, 3.0]
STD_PATH = "checkpoints/sam-3d-body-dinov3/shape_scale_std.pt"


def load_trainer(exp_dir, load_path, device):
    from sam_3d_body.configs.config import get_config_defaults
    from sam_3d_body.trainer import Trainer

    cfg = get_config_defaults()
    cfg_yaml = Path(exp_dir) / "config.yaml"
    if cfg_yaml.exists():
        cfg.merge_from_file(str(cfg_yaml))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = (
        "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )

    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(exp_dir) / "diag_tmp")).to(device)
    ckpt = torch.load(load_path, weights_only=False, map_location="cpu")
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    trainer.model.load_state_dict(sd, strict=False)
    trainer.model.eval()
    return trainer


@torch.no_grad()
def get_zero_inputs(trainer, device):
    """Return zero pose/expression inputs of the right shapes by running one forward pass."""
    loader = trainer.multiview_eval_dataloader(num_view=1, batch_size=1, dataset_name="4d-dress")
    batch = next(iter(loader))
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    bs, num_views = batch["img"].shape[:2]
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_views:
                batch[k] = v.flatten(0, 1)
    batch = trainer.preprocess(batch)
    outputs = trainer.model(batch, num_samples=1)
    mhr = outputs["mhr"]

    z = {
        "global_trans": torch.zeros_like(mhr["global_rot"][:1]),
        "global_rot": torch.zeros_like(mhr["global_rot"][:1]),
        "body_pose_params": torch.zeros_like(mhr["body_pose"][:1]),
        "hand_pose_params": torch.zeros_like(mhr["hand"][:1]),
        "expr_params": torch.zeros_like(mhr["face"][:1]),
    }
    return z


def _verts_for(mhr_head, zero_inputs, base_shape, base_scale, dim, sigma, shape_std,
               centre=True):
    shape = base_shape.clone()
    if dim is not None:
        shape[0, dim] = sigma * shape_std[dim]
    out = mhr_head.mhr_forward(
        shape_params=shape,
        scale_offsets=base_scale,
        do_pcblend=True,
        **zero_inputs,
    )
    verts = out[0] if isinstance(out, tuple) else out
    v = verts[0].cpu().numpy()
    v[:, [1, 2]] *= -1
    if centre:
        v = v - v.mean(axis=0, keepdims=True)
    return v


def render_dim_perturbation(mhr_head, renderer, zero_inputs, base_shape, base_scale,
                             dim, sigma, shape_std, device, side_view=False):
    """Render a single (dim, sigma) perturbation, front or side view."""
    verts_np = _verts_for(mhr_head, zero_inputs, base_shape, base_scale,
                          dim, sigma, shape_std)
    if side_view:
        rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=verts_np.dtype)
        verts_np = verts_np @ rot.T
    cam_t = np.array([0.0, 0.0, 3.0])
    rgba = renderer.render_rgba(
        verts_np, cam_t=cam_t, render_res=[256, 256],
        mesh_base_color=(0.85, 0.85, 0.95),
    )
    return rgba


def render_grid(dims, labels, mhr_head, renderer, zero_inputs, base_shape, base_scale,
                shape_std, device, save_path, title, side_view=False):
    n_rows = len(dims)
    n_cols = len(SIGMAS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for r, (dim, label) in enumerate(zip(dims, labels)):
        for c, s in enumerate(SIGMAS):
            rgba = render_dim_perturbation(
                mhr_head, renderer, zero_inputs, base_shape, base_scale,
                dim, s, shape_std, device, side_view=side_view,
            )
            ax = axes[r, c]
            ax.imshow(rgba)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{s:+.1f}σ" if s != 0 else "0")
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    plt.close(fig)


def render_displacement_heatmap(dims, labels, mhr_head, zero_inputs, base_shape,
                                  base_scale, shape_std, device, save_path, title):
    """Per-dim mean per-vertex displacement (mm) across body, at +1σ and +3σ.

    Shows where (in the body) each dim has the largest geometric effect.
    Uses scatter of x,y projection (front view) coloured by displacement.
    """
    # Use UNCENTRED verts so global build/scale effects are preserved.
    base_v = _verts_for(mhr_head, zero_inputs, base_shape, base_scale,
                        None, 0.0, shape_std, centre=False)
    base_v_centred = base_v - base_v.mean(axis=0, keepdims=True)

    sig_levels = [1.0, 3.0]
    n_rows = len(dims)
    n_cols = len(sig_levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, (dim, label) in enumerate(zip(dims, labels)):
        for c, s in enumerate(sig_levels):
            v = _verts_for(mhr_head, zero_inputs, base_shape, base_scale,
                           dim, s, shape_std, centre=False)
            disp_mm = np.linalg.norm(v - base_v, axis=-1) * 1000.0
            ax = axes[r, c]
            sc = ax.scatter(
                base_v_centred[:, 0], base_v_centred[:, 1],
                c=disp_mm, cmap="viridis", s=0.8, vmin=0, vmax=15,
            )
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.invert_yaxis()
            if r == 0:
                ax.set_title(f"+{s:.0f}σ")
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.text(
                0.02, 0.98,
                f"max={disp_mm.max():.0f}mm  μ={disp_mm.mean():.0f}mm",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, pad=1),
            )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="vertex displacement (mm)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"Saved {save_path}")
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.experiment_dir) / "viz_top_dims"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer(args.experiment_dir, args.load_from_ckpt, device)
    mhr_head = trainer.model.head_pose
    faces = trainer.faces

    from sam_3d_body.visualization.renderer import Renderer
    renderer = Renderer(focal_length=512, faces=faces)

    stds = torch.load(STD_PATH, weights_only=False, map_location=device)
    shape_std = stds["shape_std"].to(device).float()                # (45,)
    logger.info(
        f"shape_std: median={shape_std.median().item():.3f}, "
        f"top dims (d43..d11) σ_prior={[f'd{d}={shape_std[d].item():.3f}' for d in UNC_DIMS+DIS_DIMS]}"
    )

    zero_inputs = get_zero_inputs(trainer, device)
    B = 1
    base_shape = torch.zeros(B, 45, device=device)

    # Use mean scale (from MHR head) as the neutral base scale (68-D).
    base_scale = mhr_head.scale_mean[None, :].to(device).float()    # (1, 68)

    unc_labels = [f"d{d} (σ/σ_prior={r})" for d, r in
                  zip(UNC_DIMS, [20.6, 11.2, 6.5, 4.8, 3.2])]
    dis_labels = [f"d{d} (cv pen={p}n)" for d, p in
                  zip(DIS_DIMS, [18.7, 22.6, 9.1, 8.1, 5.5])]

    # ---- Front-view perturbation grids ----
    render_grid(UNC_DIMS, unc_labels, mhr_head, renderer, zero_inputs,
                base_shape, base_scale, shape_std, device,
                save_dir / "uncertainty_dims_front.png",
                "Top uncertainty dims — front view")
    render_grid(DIS_DIMS, dis_labels, mhr_head, renderer, zero_inputs,
                base_shape, base_scale, shape_std, device,
                save_dir / "disagreement_dims_front.png",
                "Top disagreement dims — front view")

    # ---- Side-view perturbation grids (depth-axis effects) ----
    render_grid(UNC_DIMS, unc_labels, mhr_head, renderer, zero_inputs,
                base_shape, base_scale, shape_std, device,
                save_dir / "uncertainty_dims_side.png",
                "Top uncertainty dims — side view", side_view=True)
    render_grid(DIS_DIMS, dis_labels, mhr_head, renderer, zero_inputs,
                base_shape, base_scale, shape_std, device,
                save_dir / "disagreement_dims_side.png",
                "Top disagreement dims — side view", side_view=True)

    # ---- Per-vertex displacement heatmaps (where on body) ----
    render_displacement_heatmap(UNC_DIMS, unc_labels, mhr_head, zero_inputs,
                                 base_shape, base_scale, shape_std, device,
                                 save_dir / "uncertainty_dims_disp.png",
                                 "Top uncertainty dims — per-vertex displacement")
    render_displacement_heatmap(DIS_DIMS, dis_labels, mhr_head, zero_inputs,
                                 base_shape, base_scale, shape_std, device,
                                 save_dir / "disagreement_dims_disp.png",
                                 "Top disagreement dims — per-vertex displacement")

    logger.info("Done.")


if __name__ == "__main__":
    main()
