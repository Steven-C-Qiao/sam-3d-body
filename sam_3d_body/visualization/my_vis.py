import os
from typing import Optional

import cv2

import torch
import matplotlib
import numpy as np
import pytorch_lightning as pl
from loguru import logger

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh

from sam_3d_body.metrics.metrics_tracker import scale_and_translation_transform_batch
from sam_3d_body.visualization.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
LIGHT_ORANGE = (1.0, 0.8, 0.5)
BLUE   = (0.12156863, 0.46666667, 0.70588235)   
ORANGE = (1.0,        0.49803922, 0.05490196)   
GREEN = (0.2, 1.0, 0.2)


def vis_histogram(
    merged_dists: np.ndarray,
    pred_dists: np.ndarray,
    *,
    batch_idx: int,
    save_dir: str,
) -> None:
    """
    Plot one merged distance histogram and one histogram per predicted view.

    Args:
        merged_dists: Shape (B, V) array, usually B=1 in multiview eval.
        pred_dists: Shape (N_view, V) array of per-view distances.
        batch_idx: Batch index used for file naming.
        save_dir: Directory to save the histogram image.
    """
    all_dists_for_color = np.concatenate([merged_dists.reshape(-1), pred_dists.reshape(-1)])
    max_dist = float(all_dists_for_color.max()) if all_dists_for_color.size > 0 else 0.1
    if max_dist <= 0:
        max_dist = 0.1

    bins = np.linspace(0.0, max_dist, 51)
    num_rows = 1 + pred_dists.shape[0]
    fig, axs = plt.subplots(num_rows, 1, figsize=(6, 3 * num_rows), sharex=True)
    if num_rows == 1:
        axs = [axs]

    def plot_hist_inferno(ax, data, *, title: str, alpha: float = 0.7):
        counts, edges = np.histogram(data, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Anchor at 0, exactly matching mesh color normalization.
        denom = max_dist - 0.0
        if denom <= 0:
            denom = 1.0
        normalized = np.clip((bin_centers - 0.0) / denom, 0.0, 1.0)
        cmap = plt.get_cmap("inferno")
        rgba = cmap(normalized)  # (N, 4)

        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=rgba,
            alpha=alpha,
            linewidth=0,
        )
        ax.set_title(title)
        ax.set_ylabel("Frequency")
        ax.grid(True)

    plot_hist_inferno(axs[0], merged_dists[0], title="merged_sc")
    for i in range(pred_dists.shape[0]):
        plot_hist_inferno(axs[i + 1], pred_dists[i], title=f"pred_sc{i}")

    axs[-1].set_xlabel("Distance")
    plt.tight_layout()
    hist_path = os.path.join(save_dir, f"b{batch_idx:03d}_error_hist.png")
    plt.savefig(hist_path)
    print(f"Saved histogram column to {hist_path}")
    plt.close()

def build_vertex_colors(
    dists: np.ndarray,
    *,
    min_dist: float,
    max_dist: float,
    cmap: str = "inferno",
) -> np.ndarray:
    """
    Map per-vertex distances to RGBA colors using a shared viridis scale.

    Args:
        dists: Per-vertex distances, shape (V,).
        min_dist: Global minimum distance used for normalization.
        max_dist: Global maximum distance used for normalization.
        cmap: Matplotlib colormap name, e.g. ``"viridis"``, ``"magma"``.
    """
    # For error heatmaps, we always anchor the colormap at 0.
    # `min_dist` is kept only for API compatibility.
    effective_min_dist = 0.0
    denom = max_dist - effective_min_dist
    if denom <= 0:
        denom = 1.0
    normalized = (dists - effective_min_dist) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    colors_rgb = plt.get_cmap(cmap)(normalized)[..., :3]  # (V, 3)
    vertex_colors = np.ones((colors_rgb.shape[0], 4), dtype=np.float32)
    vertex_colors[:, :3] = colors_rgb
    return vertex_colors


def build_distance_colorbar_rgb(
    *,
    min_dist: float,
    max_dist: float,
    cmap: str = "inferno",
    height: int,
    width: int = 30,
) -> np.ndarray:
    """
    Create a simple RGB colorbar image (gradient only) with numeric min/max.
    """
    # For error heatmaps, we always anchor the colorbar at 0.
    effective_min_dist = 0.0
    denom = max_dist - effective_min_dist
    if denom <= 0:
        denom = 1.0

    # Top corresponds to max distance (so "bad" = top hot color).
    values = np.linspace(max_dist, effective_min_dist, height, dtype=np.float32)
    normalized = (values - effective_min_dist) / denom
    normalized = np.clip(normalized, 0.0, 1.0)

    colors = plt.get_cmap(cmap)(normalized)[..., :3]  # (H, 3), RGB in [0,1]
    bar_rgb = (colors * 255.0).astype(np.uint8)  # (H, 3)
    bar_rgb = np.repeat(bar_rgb[:, None, :], width, axis=1)  # (H, W, 3)

    # Annotate min/max.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.7, height / 700.0))
    thickness = 1
    color_text = (0, 0, 0)  # black

    top_text = f"{max_dist:.1f}"
    bot_text = f"{0.0:.1f}"
    cv2.putText(
        bar_rgb,
        top_text,
        (2, int(12 + font_scale * 10)),
        font,
        font_scale,
        color_text,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        bar_rgb,
        bot_text,
        (2, height - int(6 + font_scale * 10)),
        font,
        font_scale,
        color_text,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return bar_rgb


def _draw_label_lines(img, lines, *, origin=(10, 10), font_scale=0.7, pad=6,
                      line_height=26, bg_alpha=0.55):
    """Draw a stack of text lines with a semi-transparent black box and white text."""
    if not lines:
        return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    widths = [cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in lines]
    box_w = max(widths) + 2 * pad
    box_h = line_height * len(lines) + 2 * pad
    x0, y0 = origin
    x1 = min(x0 + box_w, img.shape[1])
    y1 = min(y0 + box_h, img.shape[0])

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, dst=img)

    for i, text in enumerate(lines):
        y = y0 + pad + (i + 1) * line_height - 8
        cv2.putText(img, text, (x0 + pad, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def vis_prediction(img_cv2, outputs, faces, stack_vertically=True, batch=None):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    camera_center=(
        batch["cam_int"][0, 0, 2],
        batch["cam_int"][0, 1, 2],
    )

    # Get original output (mean prediction)
    mhr_outputs = outputs['mhr']
    for key in mhr_outputs:
        try:
            mhr_outputs[key] = mhr_outputs[key].cpu().detach().numpy()
        except:
            pass
    person_output = mhr_outputs

    keypoints_2d = person_output["pred_keypoints_2d"][0]
    keypoints_2d = np.concatenate(
        [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
    )

    # Get mean prediction vertices (original output)
    mean_pred_vertices = person_output["pred_vertices"][0]

    # Get samples if available
    vertex_colors = None
    if 'verts_samples' in outputs:
        verts_samples = outputs['verts_samples']
        if isinstance(verts_samples, torch.Tensor):
            verts_samples = verts_samples.cpu().detach().numpy()

        # Calculate average distance of each vertex from mean across all samples
        num_samples = verts_samples.shape[1]
        distances = []
        for i in range(num_samples):
            sample_vertices = verts_samples[0, i]
            vertex_distances = np.linalg.norm(sample_vertices - mean_pred_vertices, axis=1)
            distances.append(vertex_distances)

        avg_distances = np.mean(distances, axis=0)

        min_dist = np.min(avg_distances)
        max_dist = np.max(avg_distances)
        if max_dist > min_dist:
            normalized_distances = (avg_distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.zeros_like(avg_distances)

        viridis = cm.get_cmap('viridis')
        vertex_colors_rgb = viridis(normalized_distances)[:, :3]
        vertex_colors = np.ones((vertex_colors_rgb.shape[0], 4))
        vertex_colors[:, :3] = vertex_colors_rgb

    all_pred_vertices = person_output["pred_vertices"][0]
    all_faces = faces

    renderer = Renderer(focal_length=person_output["focal_length"][0], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            person_output["pred_cam_t"][0],
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            vertex_colors=vertex_colors,
            camera_center=camera_center,
        )
        * 255
    )

    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side = (
        renderer(
            all_pred_vertices,
            person_output["pred_cam_t"][0],
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
            vertex_colors=vertex_colors,
            camera_center=camera_center,
        )
        * 255
    )

    if stack_vertically:
        cur_img = np.concatenate([img_cv2, img_mesh, img_mesh_side], axis=0)
    else:
        cur_img = np.concatenate([img_cv2, img_mesh, img_mesh_side], axis=1)

    return cur_img


def vis_samples(
    img_cv2,
    outputs,
    faces,
    batch,
    metrics=None,
    vis_verts=None,
    stack_vertically=True,
    overlay_gt=True,
    plot_side=True,
    plot_neutral=True,
    max_sample: Optional[int] = 10,
):
    def _to_np(x):
        return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x

    b = 0  # batch element to visualise

    # ---- batch-derived quantities (full batch; index with `b` in the plotting loop) ----
    affine_all = _to_np(batch["affine_trans"]) if "affine_trans" in batch else None
    img_size_all = _to_np(batch["img_size"]) if "img_size" in batch else None
    cam_int = batch["cam_int"]
    gt_verts = _to_np(batch["vertices"])
    gt_cam_t = _to_np(batch["cam_ext"][..., :3, -1])
    gt_root_joint = _to_np(batch["joint_coords"][..., [1], :])

    affine = affine_all[b, 0] if affine_all is not None else None
    img_size = img_size_all[b, 0] if img_size_all is not None else None
    camera_center = (cam_int[b, 0, 2], cam_int[b, 1, 2])

    base_img = img_cv2.copy()
    if affine is not None:
        base_img = cv2.warpAffine(base_img.astype(np.uint8), affine, img_size)

    # ---- output-derived quantities ----
    mhr_samples = _to_np(outputs["verts_samples"])
    n_mhr = mhr_samples.shape[1]
    n_vis = min(n_mhr, max_sample) if max_sample is not None else n_mhr
    mhr_root_joint_samples = _to_np(outputs["j3d_samples"][..., 1, :])
    pred_cam_t_samples = (
        _to_np(outputs["pred_cam_t_samples"]) if "pred_cam_t_samples" in outputs else None
    )
    log_prob = None
    if "uncertainty_output" in outputs:
        log_prob = _to_np(outputs["uncertainty_output"].get("log_prob"))

    mhr_outputs = {k: _to_np(v) for k, v in outputs["mhr"].items()}
    mean_pred_vertices_np = mhr_outputs["pred_vertices"][b]

    # ---- metrics (may be None) ----
    m = metrics or {}
    mpjpe_samples = m.get("mpjpe_samples")
    mpjpe_mean = m.get("mpjpe")
    pampjpe_samples = m.get("pampjpe_samples")
    pampjpe_mean = m.get("pampjpe")
    gt_logp = m.get("gt_residual_log_prob")
    mean_logp = m.get("mean_residual_log_prob")
    kp2d_visible_samples_px = m.get("kp2d_samples_pixel_error_visible")
    kp2d_visible_mean_px = m.get("kp2d_pixel_error_visible")
    spread_invisible_samples = m.get("spread_invisible_kp3d_samples")

    # ---- per-vertex spread colours, shared inferno scale ----
    per_sample_dists = np.linalg.norm(
        mhr_samples[b, :n_vis] - mean_pred_vertices_np[None], axis=-1
    )  # (n_vis, V)
    shared_max = float(per_sample_dists.max()) if per_sample_dists.size else 1.0
    vertex_colors_samples = [
        build_vertex_colors(per_sample_dists[i], min_dist=0.0, max_dist=shared_max)
        for i in range(n_vis)
    ]
    vertex_colors_mean = build_vertex_colors(
        per_sample_dists.mean(axis=0), min_dist=0.0, max_dist=shared_max
    )

    # ---- renderers ----
    generic_camera = np.array([0.0, 0.0, 4.0])
    renderer = Renderer(focal_length=mhr_outputs["focal_length"][b], faces=faces)
    renderer_side = Renderer(focal_length=1000, faces=faces)

    # Expose as `outputs` to keep existing per-sample code below unchanged.
    outputs = mhr_outputs
    img_mesh_list = []
    img_side_list = []
    img_pa_list = []
    img_neutral_list = []
    img_neutral_sc_list = []

    # ---- precomputed aligned/neutral vertices (supplied by metrics tracker) ----
    gt_verts_b = gt_verts[b]
    gt_root = gt_root_joint[b]  # (1, 3)
    vv = vis_verts or {}
    aligned_samples = _to_np(vv["pa_sample_verts"])[b, :n_vis] if "pa_sample_verts" in vv else None
    aligned_mean = _to_np(vv["pa_mean_verts"])[b] if "pa_mean_verts" in vv else None

    neutral_available = plot_neutral and all(
        k in vv for k in ("gt_neutral_verts", "pred_neutral_verts", "sample_neutral_verts",
                          "pred_neutral_verts_sc", "sample_neutral_verts_sc")
    )
    if neutral_available:
        gt_neutral = _to_np(vv["gt_neutral_verts"])[b]
        pred_neutral = _to_np(vv["pred_neutral_verts"])[b]
        sample_neutral = _to_np(vv["sample_neutral_verts"])[b, :n_vis]
        pred_neutral_sc = _to_np(vv["pred_neutral_verts_sc"])[b]
        sample_neutral_sc = _to_np(vv["sample_neutral_verts_sc"])[b, :n_vis]
        neutral_center = gt_neutral.mean(axis=0, keepdims=True) + np.array([[0.0, 0.1, 0.0]])

        pve_samples = m.get("pve_samples")
        pve_samples = _to_np(pve_samples)[b, :n_vis] if pve_samples is not None else \
            np.linalg.norm(sample_neutral - gt_neutral[None], axis=-1).mean(axis=-1) * 1000.0
        pve_mean = m.get("pve")
        pve_mean = float(_to_np(pve_mean)[b]) if pve_mean is not None else \
            float(np.linalg.norm(pred_neutral - gt_neutral, axis=-1).mean()) * 1000.0
        pvetsc_samples = m.get("pvetsc_samples")
        pvetsc_samples = _to_np(pvetsc_samples)[b, :n_vis] if pvetsc_samples is not None else \
            np.linalg.norm(sample_neutral_sc - gt_neutral[None], axis=-1).mean(axis=-1) * 1000.0
        pvetsc_mean = m.get("pvetsc")
        pvetsc_mean = float(_to_np(pvetsc_mean)[b]) if pvetsc_mean is not None else \
            float(np.linalg.norm(pred_neutral_sc - gt_neutral, axis=-1).mean()) * 1000.0

        # ---- neutral per-vertex spread colours (shared inferno scale per row) ----
        neutral_dists = np.linalg.norm(
            sample_neutral - pred_neutral[None], axis=-1
        )  # (n_vis, V)
        neutral_max = float(neutral_dists.max()) if neutral_dists.size else 1.0
        vertex_colors_neutral_samples = [
            build_vertex_colors(neutral_dists[i], min_dist=0.0, max_dist=neutral_max)
            for i in range(n_vis)
        ]
        vertex_colors_neutral_mean = build_vertex_colors(
            neutral_dists.mean(axis=0), min_dist=0.0, max_dist=neutral_max
        )

        neutral_sc_dists = np.linalg.norm(
            sample_neutral_sc - pred_neutral_sc[None], axis=-1
        )
        neutral_sc_max = float(neutral_sc_dists.max()) if neutral_sc_dists.size else 1.0
        vertex_colors_neutral_sc_samples = [
            build_vertex_colors(neutral_sc_dists[i], min_dist=0.0, max_dist=neutral_sc_max)
            for i in range(n_vis)
        ]
        vertex_colors_neutral_sc_mean = build_vertex_colors(
            neutral_sc_dists.mean(axis=0), min_dist=0.0, max_dist=neutral_sc_max
        )
    white_bg_full = np.full_like(img_cv2, 255, dtype=np.uint8)
    if img_size is not None:
        black_bg = np.zeros((int(img_size[1]), int(img_size[0]), 3), dtype=np.uint8)
    else:
        black_bg = np.zeros_like(img_cv2)

    def _to_uint8(float_rgb):
        return (float_rgb * 255.0).clip(0, 255).astype(np.uint8)

    for i in range(n_vis):
        # ----------------------- front view -----------------------
        sample_cam_t = pred_cam_t_samples[b, i] if pred_cam_t_samples is not None else outputs["pred_cam_t"][b]
        pred_rgb = renderer(
            mhr_samples[b, i],
            sample_cam_t,
            img_cv2.copy(),
            scene_bg_color=(1, 1, 1),
            vertex_colors=vertex_colors_samples[i],
            camera_center=camera_center,
        )

        if overlay_gt:
            gt_rgba = renderer(
                gt_verts[b], gt_cam_t[b],
                white_bg_full,
                mesh_base_color=BLUE,
                scene_bg_color=(1, 1, 1),
                return_rgba=True,
                camera_center=camera_center,
            )
            alpha = gt_rgba[..., 3:4].astype(np.float32) * 0.5
            pred_rgb = alpha * gt_rgba[..., :3].astype(np.float32) + (1.0 - alpha) * pred_rgb

        img_mesh = _to_uint8(pred_rgb)

        if affine is not None:
            img_mesh = cv2.warpAffine(img_mesh, affine, img_size)

        front_lines = []
        if kp2d_visible_samples_px is not None:
            front_lines.append(f"2D err vis: {float(kp2d_visible_samples_px[b, i]):.1f} px")
        if spread_invisible_samples is not None:
            front_lines.append(
                f"3D dist to mean: {float(spread_invisible_samples[b, i]) * 1000.0:.1f} mm"
            )
        _draw_label_lines(img_mesh, front_lines)

        img_mesh_list.append(img_mesh)

        # ----------------------- side view -----------------------
        if plot_side:
            pred_side = renderer_side(
                mhr_samples[b, i] - mhr_root_joint_samples[b, i],
                generic_camera,
                black_bg,
                vertex_colors=vertex_colors_samples[i],
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=90,
            )
            gt_side = renderer_side(
                gt_verts[b] - gt_root_joint[b],
                generic_camera,
                black_bg,
                mesh_base_color=BLUE,
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=90,
                return_rgba=True,
            )
            alpha = gt_side[..., 3:4].astype(np.float32) * 0.5
            blended = alpha * gt_side[..., :3].astype(np.float32) + (1.0 - alpha) * pred_side
            img_side = _to_uint8(blended)

            side_lines = []
            if log_prob is not None:
                side_lines.append(f"log p: {float(log_prob[b, i]):.1f}")
            if mpjpe_samples is not None:
                side_lines.append(f"MPJPE: {float(mpjpe_samples[b, i]):.1f} mm")
            _draw_label_lines(img_side, side_lines)
            img_side_list.append(img_side)

            # ----------------------- PA-aligned side view -----------------------
            pa_pred_side = renderer_side(
                aligned_samples[i] - gt_root,
                generic_camera,
                black_bg,
                vertex_colors=vertex_colors_samples[i],
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=90,
            )
            gt_pa_rgba = renderer_side(
                gt_verts_b - gt_root,
                generic_camera,
                black_bg,
                mesh_base_color=BLUE,
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=90,
                return_rgba=True,
            )
            alpha_pa = gt_pa_rgba[..., 3:4].astype(np.float32) * 0.5
            blended_pa = alpha_pa * gt_pa_rgba[..., :3].astype(np.float32) + (1.0 - alpha_pa) * pa_pred_side
            img_pa = _to_uint8(blended_pa)

            pa_lines = []
            if pampjpe_samples is not None:
                pa_lines.append(f"PA-MPJPE: {float(pampjpe_samples[b, i]):.1f} mm")
            _draw_label_lines(img_pa, pa_lines)
            img_pa_list.append(img_pa)

            # ----------------------- neutral raw -----------------------
            if neutral_available:
                pred_n_side = renderer_side(
                    sample_neutral[i] - neutral_center,
                    generic_camera,
                    black_bg,
                    vertex_colors=vertex_colors_neutral_samples[i],
                    scene_bg_color=(0, 0, 0),
                    side_view=True,
                    rot_angle=0,
                )
                gt_n_rgba = renderer_side(
                    gt_neutral - neutral_center,
                    generic_camera,
                    black_bg,
                    mesh_base_color=BLUE,
                    scene_bg_color=(0, 0, 0),
                    side_view=True,
                    rot_angle=0,
                    return_rgba=True,
                )
                alpha_n = gt_n_rgba[..., 3:4].astype(np.float32) * 0.5
                blended_n = alpha_n * gt_n_rgba[..., :3].astype(np.float32) + (1.0 - alpha_n) * pred_n_side
                img_n = _to_uint8(blended_n)
                _draw_label_lines(img_n, [f"PVE: {float(pve_samples[i]):.1f} mm"])
                img_neutral_list.append(img_n)

                # ----------------------- neutral scale+trans aligned -----------------------
                pred_nsc_side = renderer_side(
                    sample_neutral_sc[i] - neutral_center,
                    generic_camera,
                    black_bg,
                    vertex_colors=vertex_colors_neutral_sc_samples[i],
                    scene_bg_color=(0, 0, 0),
                    side_view=True,
                    rot_angle=0,
                )
                gt_nsc_rgba = renderer_side(
                    gt_neutral - neutral_center,
                    generic_camera,
                    black_bg,
                    mesh_base_color=BLUE,
                    scene_bg_color=(0, 0, 0),
                    side_view=True,
                    rot_angle=0,
                    return_rgba=True,
                )
                alpha_nsc = gt_nsc_rgba[..., 3:4].astype(np.float32) * 0.5
                blended_nsc = alpha_nsc * gt_nsc_rgba[..., :3].astype(np.float32) + (1.0 - alpha_nsc) * pred_nsc_side
                img_nsc = _to_uint8(blended_nsc)
                _draw_label_lines(img_nsc, [f"PVE-T-SC: {float(pvetsc_samples[i]):.1f} mm"])
                img_neutral_sc_list.append(img_nsc)

    axis = 0 if stack_vertically else 1
    img_mesh_list = np.concatenate(img_mesh_list, axis=axis)
    img_side_list = np.concatenate(img_side_list, axis=axis)
    if img_pa_list:
        img_pa_list = np.concatenate(img_pa_list, axis=axis)
    else:
        img_pa_list = None
    if img_neutral_list:
        img_neutral_list = np.concatenate(img_neutral_list, axis=axis)
        img_neutral_sc_list = np.concatenate(img_neutral_sc_list, axis=axis)
    else:
        img_neutral_list = None
        img_neutral_sc_list = None

    # ----------------------- Top-left -----------------------
    if overlay_gt:
        gt_base_rgba = renderer(
            gt_verts[b],
            gt_cam_t[b],
            img_cv2.copy(),
            mesh_base_color=BLUE,
            scene_bg_color=(1, 1, 1),
            return_rgba=True,
            camera_center=camera_center,
        )

        mean_pred_verts = outputs["pred_vertices"][b]
        mean_pred_cam_t = outputs["pred_cam_t"][b]
        mean_pred_root_joint = outputs["pred_joint_coords"][b][..., [1], :]

        mean_pred_rgb_full = renderer(
            mean_pred_verts,
            mean_pred_cam_t,
            img_cv2.copy(),
            scene_bg_color=(1, 1, 1),
            vertex_colors=vertex_colors_mean,
            camera_center=camera_center,
        )
        alpha = gt_base_rgba[..., 3:4].astype(np.float32) * 0.5
        blended_front = alpha * gt_base_rgba[..., :3].astype(np.float32) + (1.0 - alpha) * mean_pred_rgb_full
        gt_base_img = _to_uint8(blended_front)

        # ----------------------- Bottom-left -----------------------
        mean_pred_unc = renderer_side(
            mean_pred_verts - mean_pred_root_joint,
            generic_camera,
            black_bg,
            scene_bg_color=(0, 0, 0),
            vertex_colors=vertex_colors_mean,
            side_view=True,
            rot_angle=90,
        )
        gt_rgba_unc = renderer_side(
            gt_verts[b] - gt_root_joint[b],
            generic_camera,
            black_bg,
            mesh_base_color=BLUE,
            scene_bg_color=(0, 0, 0),
            side_view=True,
            rot_angle=90,
            return_rgba=True,
        )
        alpha_unc = gt_rgba_unc[..., 3:4].astype(np.float32) * 0.5
        blended_unc = alpha_unc * gt_rgba_unc[..., :3].astype(np.float32) + (1.0 - alpha_unc) * mean_pred_unc
        mean_unc_panel = _to_uint8(blended_unc)


        mean_side_lines = []
        if mean_logp is not None:
            mean_side_lines.append(f"log p (mean): {float(mean_logp[b]):.1f}")
        if mpjpe_mean is not None:
            mean_side_lines.append(f"MPJPE (mean): {float(mpjpe_mean[b]):.1f} mm")
        if gt_logp is not None:
            mean_side_lines.append(f"log p (gt residual): {float(gt_logp[b]):.1f}")
        _draw_label_lines(mean_unc_panel, mean_side_lines)

        # ----------------------- Bottom-left (PA-aligned) -----------------------
        mean_pa_pred = renderer_side(
            aligned_mean - gt_root,
            generic_camera,
            black_bg,
            vertex_colors=vertex_colors_mean,
            scene_bg_color=(0, 0, 0),
            side_view=True,
            rot_angle=90,
        )
        gt_pa_rgba_mean = renderer_side(
            gt_verts_b - gt_root,
            generic_camera,
            black_bg,
            mesh_base_color=BLUE,
            scene_bg_color=(0, 0, 0),
            side_view=True,
            rot_angle=90,
            return_rgba=True,
        )
        alpha_mpa = gt_pa_rgba_mean[..., 3:4].astype(np.float32) * 0.5
        blended_mpa = alpha_mpa * gt_pa_rgba_mean[..., :3].astype(np.float32) + (1.0 - alpha_mpa) * mean_pa_pred
        mean_pa_panel = _to_uint8(blended_mpa)

        mean_pa_lines = []
        if pampjpe_mean is not None:
            mean_pa_lines.append(f"PA-MPJPE (mean): {float(pampjpe_mean[b]):.1f} mm")
        _draw_label_lines(mean_pa_panel, mean_pa_lines)

        # ----------------------- mean neutral panels -----------------------
        mean_neutral_panel = None
        mean_neutral_sc_panel = None
        if neutral_available:
            pred_n_mean = renderer_side(
                pred_neutral - neutral_center,
                generic_camera,
                black_bg,
                vertex_colors=vertex_colors_neutral_mean,
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=0,
            )
            gt_n_rgba_mean = renderer_side(
                gt_neutral - neutral_center,
                generic_camera,
                black_bg,
                mesh_base_color=BLUE,
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=0,
                return_rgba=True,
            )
            alpha_nm = gt_n_rgba_mean[..., 3:4].astype(np.float32) * 0.5
            blended_nm = alpha_nm * gt_n_rgba_mean[..., :3].astype(np.float32) + (1.0 - alpha_nm) * pred_n_mean
            mean_neutral_panel = _to_uint8(blended_nm)
            _draw_label_lines(mean_neutral_panel, [f"PVE (mean): {pve_mean:.1f} mm"])

            pred_nsc_mean = renderer_side(
                pred_neutral_sc - neutral_center,
                generic_camera,
                black_bg,
                vertex_colors=vertex_colors_neutral_sc_mean,
                scene_bg_color=(0, 0, 0),
                side_view=True,
                rot_angle=0,
            )
            alpha_nscm = gt_n_rgba_mean[..., 3:4].astype(np.float32) * 0.5
            blended_nscm = alpha_nscm * gt_n_rgba_mean[..., :3].astype(np.float32) + (1.0 - alpha_nscm) * pred_nsc_mean
            mean_neutral_sc_panel = _to_uint8(blended_nscm)
            _draw_label_lines(mean_neutral_sc_panel, [f"PVE-T-SC (mean): {pvetsc_mean:.1f} mm"])

        if affine is not None:
            gt_base_img = cv2.warpAffine(gt_base_img, affine, img_size)

        mean_front_lines = []
        if kp2d_visible_mean_px is not None:
            mean_front_lines.append(
                f"2D err vis (mean): {float(kp2d_visible_mean_px[b]):.1f} px"
            )
        _draw_label_lines(gt_base_img, mean_front_lines)
    else:
        gt_base_img = base_img
        mean_unc_panel = np.zeros_like(base_img)
        mean_pa_panel = np.zeros_like(base_img)
        mean_neutral_panel = None
        mean_neutral_sc_panel = None

    # Build each row and (when rows are horizontal) attach a per-row colorbar.
    # dists are in metres; colorbars are labelled in mm.
    attach_cbar = not stack_vertically

    def _with_cbar(row_img, max_dist_m):
        if not attach_cbar:
            return row_img
        h = row_img.shape[0]
        cbar = build_distance_colorbar_rgb(
            min_dist=0.0,
            max_dist=float(max_dist_m) * 1000.0,
            height=h,
            width=60,
        )
        return np.concatenate([row_img, cbar], axis=1)

    rows = []
    rows.append(_with_cbar(
        np.concatenate([gt_base_img, img_mesh_list], axis=axis), shared_max
    ))
    rows.append(_with_cbar(
        np.concatenate([mean_unc_panel, img_side_list], axis=axis), shared_max
    ))
    if img_neutral_list is not None and mean_neutral_panel is not None:
        rows.append(_with_cbar(
            np.concatenate([mean_neutral_panel, img_neutral_list], axis=axis),
            neutral_max,
        ))
    if img_pa_list is not None:
        rows.append(_with_cbar(
            np.concatenate([mean_pa_panel, img_pa_list], axis=axis), shared_max
        ))
    if img_neutral_sc_list is not None and mean_neutral_sc_panel is not None:
        rows.append(_with_cbar(
            np.concatenate([mean_neutral_sc_panel, img_neutral_sc_list], axis=axis),
            neutral_sc_max,
        ))

    cur_img = np.concatenate(rows, axis=1 - axis)
    return cur_img


class Visualiser(pl.LightningModule):
    def __init__(self, save_dir, cfg=None, rank=0, faces=None, max_plots=None):
        super().__init__()
        self.save_dir = save_dir
        self.rank = rank
        self.cfg = cfg
        self._suffix = ""
        self.faces = faces  # Store faces for mesh rendering
        self.max_plots = max_plots

    def set_global_rank(self, global_rank):
        self.rank = global_rank

    def _get_filename(self, suffix=""):
        """
        Generate filename with unified format:
        - Train:  ep_xxx_train_xxxxxx{suffix}.png
        - Val:    ep_xxx_val[_dataset]{suffix}.png
        """
        epoch_part = f"ep_{self._epoch:03d}"
        if self._split == "train":
            step_part = f"{self.counter:06d}" if self.counter is not None else "000000"
            return f"{epoch_part}_train_{step_part}{suffix}.png"
        elif self._split.startswith("val"):
            # self._split is e.g. "val_dataset" or just "val"
            return f"{epoch_part}_{self._split}{suffix}.png"
        else:
            # Fallback
            step_part = f"{self.counter:06d}" if self.counter is not None else "000000"
            split_part = f"_{self._split}" if self._split else ""
            return f"{epoch_part}{split_part}_{step_part}{suffix}.png"

    def visualise(
        self,
        predictions,
        batch,
        batch_idx=None,
        split=None,
        epoch=None,
        global_step=None,
    ):

        if self.rank != 0:
            return None

        # set suffix for this visualisation pass
        self._suffix = f"_{epoch}_{split}" if epoch is not None and split else ""
        # Store epoch and split separately for file naming
        self._epoch = epoch if epoch is not None else 0
        self._split = split if split else ""

        self.counter = global_step

        # Convert predictions to numpy if tensor
        for k, v in predictions.items():
            predictions[k] = (
                v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
            )
        for k, v in batch.items():
            batch[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v

        # self.visualise_ray_debug(predictions, batch)

        batch["keypoints_3d"][..., [1, 2]] *= -1
        predictions["mhr"]["pred_keypoints_3d"][..., [1, 2]] *= -1
        predictions["kp3d_samples"][..., [1, 2]] *= -1
        predictions["verts_samples"][..., [1, 2]] *= -1
        predictions["mhr"]["pred_vertices"][..., [1, 2]] *= -1

        # self.visualise_keypoints_3d(predictions, batch)

        # self.visualise_2d_keypoints_full(predictions, batch)
        self.visualise_2d_keypoints_cropped(predictions, batch)

        # self.visualise_mesh(predictions, batch)
        # self.visualise_mesh_pyplot(predictions, batch)

    def visualise_ray_debug(self, predictions, batch):
        """
        Debug plot: camera rays and NF keypoint samples in the (unflipped) body frame.

        Called BEFORE visualise_full so the keypoints are still in their original
        coordinate frame, consistent with batch["cam_ext"] / batch["trans_cam"].

        Left panel — 3D scene:
          • Camera centre (black star)
          • GT keypoints: blue = visible, red = invisible
          • Thin rays from camera centre through each visible GT keypoint
          • NF samples per joint: orange = visible, salmon = invisible

        Right panel — along-ray vs perp-ray scatter (visible joints only):
          x-axis = along-ray displacement from sample mean  (depth diversity, want large)
          y-axis = ⊥-ray displacement from sample mean    (2D inconsistency, want small)
        """
        if "kp3d_samples" not in predictions:
            return

        gt_kp3d = batch["keypoints_3d"][0]        # (J, 3) — unflipped body frame
        visibility = batch["visibility"][0].astype(bool)  # (J,)
        samples = predictions["kp3d_samples"][0]   # (N, J, 3)

        if "cam_ext" in batch:
            trans_cam = batch["cam_ext"][0, :3, 3]  # (3,)
        elif "trans_cam" in batch:
            trans_cam = batch["trans_cam"][0]        # (3,)
        else:
            return  # no camera info available

        # Camera centre in body frame: P_cam = P_body + trans_cam → cam at -trans_cam
        cam_center = -trans_cam  # (3,)

        # Unit ray from camera centre through each GT joint (in body frame)
        gt_kp3d_cam = gt_kp3d + trans_cam           # (J, 3) — camera-space coords
        ray_norms = np.linalg.norm(gt_kp3d_cam, axis=-1, keepdims=True)
        rays = gt_kp3d_cam / (ray_norms + 1e-8)     # (J, 3) unit vectors

        N = samples.shape[0]

        # Decompose sample spread into along-ray and perpendicular components
        sample_mean = samples.mean(axis=0)           # (J, 3)
        centered = samples - sample_mean[None]        # (N, J, 3)
        scalar_proj = (centered * rays[None]).sum(axis=-1)          # (N, J)
        perp_vec = centered - scalar_proj[:, :, None] * rays[None]  # (N, J, 3)
        perp_mag = np.linalg.norm(perp_vec, axis=-1)                # (N, J)

        fig = plt.figure(figsize=(18, 8))

        # ---- LEFT: 3D scene ----
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")

        # GT vertices as a faint grey body silhouette
        if "vertices" in batch:
            verts = batch["vertices"][0]  # (V, 3)
            # Subsample for speed — 2000 points is plenty
            idx = np.random.choice(len(verts), size=min(20000, len(verts)), replace=False)
            vx = verts[idx]
            ax3d.scatter(
                vx[:, 0], vx[:, 1], vx[:, 2],
                color="lightgray", s=1, alpha=0.15, zorder=1,
            )

        ax3d.scatter(*cam_center, color="black", s=200, marker="*", zorder=10)

        for j in range(len(visibility)):
            kp = gt_kp3d[j]
            samps_j = samples[:, j, :]
            if visibility[j]:
                ax3d.scatter(*kp, color="blue", s=25, alpha=0.9, zorder=5)
                # Ray from camera through GT joint, extended 15 % beyond
                t = np.linspace(0.0, 1.15, 25)
                ray_pts = cam_center + np.outer(t, kp - cam_center)
                ax3d.plot(
                    ray_pts[:, 0], ray_pts[:, 1], ray_pts[:, 2],
                    color="cornflowerblue", alpha=0.2, linewidth=0.6,
                )
                ax3d.scatter(
                    samps_j[:, 0], samps_j[:, 1], samps_j[:, 2],
                    color="orange", s=4, alpha=0.25,
                )
            else:
                ax3d.scatter(*kp, color="red", s=25, alpha=0.9, zorder=5)
                ax3d.scatter(
                    samps_j[:, 0], samps_j[:, 1], samps_j[:, 2],
                    color="salmon", s=4, alpha=0.25,
                )

        # Legend proxies
        ax3d.scatter([], [], color="black", s=80, marker="*", label="Camera")
        ax3d.scatter([], [], color="lightgray", s=20, label="GT vertices")
        ax3d.scatter([], [], color="blue", s=20, label="GT visible")
        ax3d.scatter([], [], color="red", s=20, label="GT invisible")
        ax3d.scatter([], [], color="orange", s=15, label="Samples (vis. joint)")
        ax3d.scatter([], [], color="salmon", s=15, label="Samples (invis. joint)")
        ax3d.plot([], [], color="cornflowerblue", linewidth=1, label="Camera ray")
        ax3d.legend(fontsize=7)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title("Camera rays & NF samples (body frame)")
        ax3d.view_init(elev=-5, azim=10 + 180, vertical_axis="y")

        # Fit view to the human body only (not the distant camera centre).
        # Rays are clipped naturally at the axis limits.
        body_pts = gt_kp3d
        mid = (body_pts.max(axis=0) + body_pts.min(axis=0)) / 2.0
        half = (body_pts.max(axis=0) - body_pts.min(axis=0)).max() / 2.0 + 0.3
        ax3d.set_xlim(mid[0] - half, mid[0] + half)
        ax3d.set_ylim(mid[1] - half, mid[1] + half)
        ax3d.set_zlim(mid[2] - half, mid[2] + half)
        ax3d.invert_yaxis()

        # ---- RIGHT: along-ray vs perp scatter ----
        ax2d = fig.add_subplot(1, 2, 2)
        vis_indices = np.where(visibility)[0]
        cmap_tab = plt.get_cmap("tab20")

        for ci, j in enumerate(vis_indices):
            ax2d.scatter(
                scalar_proj[:, j],
                perp_mag[:, j],
                color=cmap_tab(ci % 20),
                s=10,
                alpha=0.5,
                label=f"j{j}",
            )

        ax2d.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax2d.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax2d.set_xlabel("Along-ray displacement  (depth diversity  ↔  want large)")
        ax2d.set_ylabel("⊥-ray displacement  (2D inconsistency  ↑  want small)")
        ax2d.set_title("Sample spread decomposition — visible joints")
        if len(vis_indices) <= 15:
            ax2d.legend(fontsize=6, ncol=2)

        plt.tight_layout()
        filename = self._get_filename("_ray_debug")
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, filename), dpi=120, bbox_inches="tight")
        plt.close()

    def visualise_keypoints_3d(self, predictions, batch):
        """
        Generate 3D scatter plots visualizing GT, predicted, and sample keypoints.

        Args:
            predictions: Dictionary containing model predictions
            batch: Dictionary containing batch data including ground truth
        """
        # Extract keypoints from batch and predictions
        gt_keypoints_3d = batch["keypoints_3d"]
        pred_keypoints_3d = predictions["mhr"]["pred_keypoints_3d"]  # (B, 70, 3)
        pred_keypoints_3d_samples = predictions.get("kp3d_samples", None)

        # Handle different input shapes
        if gt_keypoints_3d.ndim == 4:  # (B, N, 70, 3)
            gt_kp = gt_keypoints_3d[0, 0]  # First batch, first person
        elif gt_keypoints_3d.ndim == 3:  # (B, 70, 3)
            gt_kp = gt_keypoints_3d[0]  # First batch
        else:
            gt_kp = gt_keypoints_3d[0] if gt_keypoints_3d.ndim > 1 else gt_keypoints_3d

        # Get first batch prediction
        if pred_keypoints_3d.ndim == 3:  # (B, 70, 3)
            pred_kp = pred_keypoints_3d[0]  # First batch
        else:
            pred_kp = pred_keypoints_3d

        # Get sample keypoints if available
        sample_kps = None
        if pred_keypoints_3d_samples is not None:
            if isinstance(pred_keypoints_3d_samples, torch.Tensor):
                pred_keypoints_3d_samples = (
                    pred_keypoints_3d_samples.cpu().detach().numpy()
                )
            # Shape: (B, num_samples, 70, 3)
            if pred_keypoints_3d_samples.ndim == 4:
                sample_kps = pred_keypoints_3d_samples[0]  # (num_samples, 70, 3)
            else:
                sample_kps = pred_keypoints_3d_samples

        # Ensure numpy arrays
        if isinstance(gt_kp, torch.Tensor):
            gt_kp = gt_kp.cpu().detach().numpy()
        if isinstance(pred_kp, torch.Tensor):
            pred_kp = pred_kp.cpu().detach().numpy()

        # Determine number of subplots
        num_samples = sample_kps.shape[0] if sample_kps is not None else 0
        if self.max_plots is not None:
            num_samples = min(num_samples, self.max_plots)
        num_cols = 3 + num_samples  # GT, Pred, Overlay, Samples

        # Create figure with subplots
        fig = plt.figure(figsize=(8 * num_cols, 8))

        # Left subplot: Ground Truth
        ax1 = fig.add_subplot(1, num_cols, 1, projection="3d")
        ax1.scatter(
            gt_kp[:, 0],
            gt_kp[:, 1],
            gt_kp[:, 2],
            c="blue",
            marker="o",
            s=50,
            alpha=0.6,
            label="GT Keypoints",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Ground Truth 3D Keypoints")
        ax1.legend()
        ax1.grid(True)

        # Set equal aspect ratio for better visualization
        max_range = (
            np.array(
                [
                    gt_kp[:, 0].max() - gt_kp[:, 0].min(),
                    gt_kp[:, 1].max() - gt_kp[:, 1].min(),
                    gt_kp[:, 2].max() - gt_kp[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (gt_kp[:, 0].max() + gt_kp[:, 0].min()) * 0.5
        mid_y = (gt_kp[:, 1].max() + gt_kp[:, 1].min()) * 0.5
        mid_z = (gt_kp[:, 2].max() + gt_kp[:, 2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        ax1.view_init(elev=10, azim=20, vertical_axis="y")

        # Middle subplot: Predicted
        ax2 = fig.add_subplot(1, num_cols, 2, projection="3d")
        ax2.scatter(
            pred_kp[:, 0],
            pred_kp[:, 1],
            pred_kp[:, 2],
            c="red",
            marker="^",
            s=50,
            alpha=0.6,
            label="Predicted Keypoints",
        )
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_title("Predicted 3D Keypoints")
        ax2.legend()
        ax2.grid(True)

        # Set equal aspect ratio for better visualization
        max_range_pred = (
            np.array(
                [
                    pred_kp[:, 0].max() - pred_kp[:, 0].min(),
                    pred_kp[:, 1].max() - pred_kp[:, 1].min(),
                    pred_kp[:, 2].max() - pred_kp[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x_pred = (pred_kp[:, 0].max() + pred_kp[:, 0].min()) * 0.5
        mid_y_pred = (pred_kp[:, 1].max() + pred_kp[:, 1].min()) * 0.5
        mid_z_pred = (pred_kp[:, 2].max() + pred_kp[:, 2].min()) * 0.5
        ax2.set_xlim(mid_x_pred - max_range_pred, mid_x_pred + max_range_pred)
        ax2.set_ylim(mid_y_pred - max_range_pred, mid_y_pred + max_range_pred)
        ax2.set_zlim(mid_z_pred - max_range_pred, mid_z_pred + max_range_pred)
        ax2.view_init(elev=10, azim=20, vertical_axis="y")

        # Third subplot: Overlay GT and Predicted
        ax3 = fig.add_subplot(1, num_cols, 3, projection="3d")
        ax3.scatter(
            gt_kp[:, 0],
            gt_kp[:, 1],
            gt_kp[:, 2],
            c="blue",
            marker="o",
            s=40,
            alpha=0.6,
            label="GT",
        )
        ax3.scatter(
            pred_kp[:, 0],
            pred_kp[:, 1],
            pred_kp[:, 2],
            c="red",
            marker="^",
            s=40,
            alpha=0.6,
            label="Pred",
        )
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.set_title("Overlay: GT vs Pred")
        ax3.legend()
        ax3.grid(True)

        # Use combined range for overlay subplot
        all_pts = np.concatenate([gt_kp, pred_kp], axis=0)
        max_range_ov = (
            np.array(
                [
                    all_pts[:, 0].max() - all_pts[:, 0].min(),
                    all_pts[:, 1].max() - all_pts[:, 1].min(),
                    all_pts[:, 2].max() - all_pts[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x_ov = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
        mid_y_ov = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
        mid_z_ov = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5
        ax3.set_xlim(mid_x_ov - max_range_ov, mid_x_ov + max_range_ov)
        ax3.set_ylim(mid_y_ov - max_range_ov, mid_y_ov + max_range_ov)
        ax3.set_zlim(mid_z_ov - max_range_ov, mid_z_ov + max_range_ov)
        ax3.view_init(elev=10, azim=20, vertical_axis="y")

        # Plot sample keypoints if available
        if sample_kps is not None:
            for i in range(num_samples):
                sample_kp = sample_kps[i]  # (70, 3)
                ax = fig.add_subplot(1, num_cols, 4 + i, projection="3d")
                ax.scatter(
                    sample_kp[:, 0],
                    sample_kp[:, 1],
                    sample_kp[:, 2],
                    c="green",
                    marker="s",
                    s=50,
                    alpha=0.6,
                    label=f"Sample {i+1}",
                )
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Sample {i+1} Keypoints")
                ax.legend()
                ax.grid(True)

                # Set equal aspect ratio
                max_range_sample = (
                    np.array(
                        [
                            sample_kp[:, 0].max() - sample_kp[:, 0].min(),
                            sample_kp[:, 1].max() - sample_kp[:, 1].min(),
                            sample_kp[:, 2].max() - sample_kp[:, 2].min(),
                        ]
                    ).max()
                    / 2.0
                )
                mid_x_sample = (sample_kp[:, 0].max() + sample_kp[:, 0].min()) * 0.5
                mid_y_sample = (sample_kp[:, 1].max() + sample_kp[:, 1].min()) * 0.5
                mid_z_sample = (sample_kp[:, 2].max() + sample_kp[:, 2].min()) * 0.5
                ax.set_xlim(
                    mid_x_sample - max_range_sample, mid_x_sample + max_range_sample
                )
                ax.set_ylim(
                    mid_y_sample - max_range_sample, mid_y_sample + max_range_sample
                )
                ax.set_zlim(
                    mid_z_sample - max_range_sample, mid_z_sample + max_range_sample
                )
                ax.view_init(elev=10, azim=20, vertical_axis="y")

        plt.tight_layout()

        # Save the figure
        filename = self._get_filename("_keypoints_3d")
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualise_mesh_pyplot(self, predictions, batch):
        """
        Visualize GT vertices, predicted vertices, and sample vertices as 3D scatter plots.
        Mimics visualise_mesh but uses matplotlib 3D scatter plots instead of pyrender.

        Args:
            predictions: Dict with 'mhr' containing predictions and 'mhr_samples' containing samples
            batch: Dict with 'vertices' (GT) and other batch data
        """
        # Extract data for first batch item
        gt_verts = batch["vertices"][0]  # (18439, 3)
        pred_verts = predictions["mhr"]["pred_vertices"][0]  # (18439, 3)
        pred_verts_samples = predictions["verts_samples"][0]  # (num_samples, 18439, 3)

        # Convert to numpy if tensor
        if isinstance(gt_verts, torch.Tensor):
            gt_verts = gt_verts.cpu().detach().numpy()
        if isinstance(pred_verts, torch.Tensor):
            pred_verts = pred_verts.cpu().detach().numpy()
        if isinstance(pred_verts_samples, torch.Tensor):
            pred_verts_samples = pred_verts_samples.cpu().detach().numpy()

        # Create figure with subplots
        num_samples = pred_verts_samples.shape[0]
        if self.max_plots is not None:
            num_samples = min(num_samples, self.max_plots)
        num_cols = 3 + num_samples  # GT, Pred, Samples
        fig = plt.figure(figsize=(6 * num_cols, 6))

        # Helper function to set equal aspect ratio and view
        def set_3d_axes_equal(ax, vertices, title, color, marker="o"):
            """Set equal aspect ratio and consistent view for 3D plot."""
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                c=color,
                marker=marker,
                s=1,
                alpha=0.5,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(title)
            ax.grid(True)

            # Set equal aspect ratio
            max_range = (
                np.array(
                    [
                        vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min(),
                        vertices[:, 2].max() - vertices[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.view_init(elev=10, azim=20, vertical_axis="y")

        # Plot GT vertices
        ax1 = fig.add_subplot(1, num_cols, 1, projection="3d")
        set_3d_axes_equal(ax1, gt_verts, "GT Vertices", "blue", marker="o")

        # Plot predicted vertices
        ax2 = fig.add_subplot(1, num_cols, 2, projection="3d")
        set_3d_axes_equal(ax2, pred_verts, "Predicted Vertices", "red", marker="^")

        # Plot sample vertices
        for i in range(num_samples):
            sample_verts = pred_verts_samples[i]  # (18439, 3)
            ax = fig.add_subplot(1, num_cols, 3 + i, projection="3d")
            set_3d_axes_equal(
                ax, sample_verts, f"Sample {i+1} Vertices", "green", marker="s"
            )

        plt.tight_layout()

        # Save the figure
        filename = self._get_filename("_mesh_vertices_3d")
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualise_mesh(self, predictions, batch):
        """
        Render GT vertices, predicted vertices, and sample vertices in separate subplots.

        Args:
            predictions: Dict with 'mhr' containing predictions and 'mhr_samples' containing samples
            batch: Dict with 'vertices' (GT) and other batch data
        """
        # Extract data for first batch item
        gt_verts = batch["vertices"][0]  # (18439, 3)
        pred_verts = predictions["mhr"]["pred_vertices"][0]  # (18439, 3)
        pred_verts_samples = predictions["verts_samples"][0]  # (num_samples, 18439, 3)

        faces = self.faces

        # Get camera parameters
        if "focal_length" in batch:
            focal_length = batch["focal_length"][0]
            if isinstance(focal_length, torch.Tensor):
                focal_length = focal_length.cpu().numpy()
            if focal_length.ndim > 0:
                focal_length = focal_length[0]  # Use first focal length
        else:
            # Default focal length
            focal_length = 5000.0

        # Camera translation - center the mesh
        # Compute center and scale of vertices for camera positioning
        verts_center = np.mean(gt_verts, axis=0)
        verts_scale = np.max(gt_verts, axis=0) - np.min(gt_verts, axis=0)
        # Center the vertices using verts_center
        gt_verts = gt_verts - verts_center
        pred_verts = pred_verts - verts_center
        pred_verts_samples = pred_verts_samples - verts_center  # (num_samples, V, 3)
        # Place camera at a distance proportional to mesh size
        cam_distance = np.max(verts_scale) * 2.5
        cam_t = np.array([0, 0, cam_distance])

        # Render size
        render_size = (512, 512)

        # Create figure with subplots
        num_samples = pred_verts_samples.shape[0]
        if self.max_plots is not None:
            num_samples = min(num_samples, self.max_plots)
        num_cols = 3 + num_samples  # GT, Pred, Samples
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
        if num_cols == 1:
            axes = [axes]

        # Render GT mesh
        gt_img = self._render_mesh(
            gt_verts,
            faces,
            cam_t,
            focal_length,
            render_size,
            mesh_color=(0.2, 0.6, 0.8),
        )
        axes[0].imshow(gt_img)
        axes[0].set_title("GT Mesh")
        axes[0].axis("off")

        # Render predicted mesh
        pred_img = self._render_mesh(
            pred_verts,
            faces,
            cam_t,
            focal_length,
            render_size,
            mesh_color=(0.8, 0.2, 0.2),
        )
        axes[1].imshow(pred_img)
        axes[1].set_title("Predicted Mesh")
        axes[1].axis("off")

        # Render sample meshes
        for i in range(num_samples):
            sample_verts = pred_verts_samples[i]  # (18439, 3)
            sample_img = self._render_mesh(
                sample_verts,
                faces,
                cam_t,
                focal_length,
                render_size,
                mesh_color=(0.2, 0.8, 0.4),
            )
            axes[2 + i].imshow(sample_img)
            axes[2 + i].set_title(f"Sample {i+1}")
            axes[2 + i].axis("off")

        plt.tight_layout()

        # Save the figure
        filename = self._get_filename("_mesh_comparison")
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _render_mesh(
        self,
        vertices,
        faces,
        cam_t,
        focal_length,
        render_size,
        mesh_color=(0.8, 0.8, 0.8),
    ):
        """
        Render a mesh using pyrender.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            cam_t: (3,) camera translation
            focal_length: scalar focal length
            render_size: (H, W) render resolution
            mesh_color: (R, G, B) mesh color in [0, 1]

        Returns:
            Rendered image as numpy array (H, W, 3) in [0, 255]
        """
        # Convert to numpy if tensor
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().detach().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().detach().numpy()
        if isinstance(cam_t, torch.Tensor):
            cam_t = cam_t.cpu().detach().numpy()

        h, w = render_size

        # Create renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

        # Create mesh
        if faces is None:
            # If no faces provided, create a simple mesh (this is a fallback)
            logger.warning("No faces provided, creating placeholder mesh")
            # For now, return a blank image
            return np.ones((h, w, 3), dtype=np.uint8) * 255

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())

        # Apply standard rotation (180 degrees around X axis)
        # rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # mesh.apply_transform(rot)

        # Create material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(mesh_color[0], mesh_color[1], mesh_color[2], 1.0),
        )

        # Create pyrender mesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # Create scene
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 0.0],  # White background
            ambient_light=(0.3, 0.3, 0.3),
        )
        scene.add(pyrender_mesh, "mesh")

        # Setup camera
        camera_pose = np.eye(4)
        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0  # Flip X for pyrender
        camera_pose[:3, 3] = camera_translation

        camera_center = [w / 2.0, h / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        # Add lights
        light_nodes = self._create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        # Render
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        # Convert to [0, 255] uint8
        color = color.astype(np.float32) / 255.0
        # Extract RGB and convert to uint8
        rgb = (color[:, :, :3] * 255).astype(np.uint8)

        return rgb

    def _create_raymond_lights(self):
        """Create raymond lights for rendering."""
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )

        return nodes

    def visualise_2d_keypoints_full(self, predictions, batch):
        """
        Visualize ground truth, predicted mean, and sampled 2D keypoints on the full original image.

        Args:
            predictions: Model predictions dictionary (already converted to numpy)
            batch: Input batch dictionary (already converted to numpy)
        """
        if "kp2d_samples" not in predictions:
            logger.warning(
                "No sample keypoints found in predictions. Skipping 2D keypoint visualization on full image."
            )
            return

        # For consistency with other visualisation functions, only visualise the first element in the batch
        batch_idx = 0

        # Get full original image
        image_original = batch["img_ori"][batch_idx]  # (H, W, 3) e.g., (720, 1280, 3)
        if image_original.max() <= 1.0:
            image_original = (image_original * 255).astype(np.uint8)
        else:
            image_original = image_original.astype(np.uint8)

        # Get predicted keypoints in full image coordinates
        pred_kp2d_full = predictions["mhr"]["pred_keypoints_2d"][
            batch_idx
        ]  # [70, 2] in original pixel coords

        # Get GT keypoints - they are now normalized to [-0.5, 0.5] in cropped coordinate space
        gt_kp2d_normalized = batch["keypoints_2d"][
            batch_idx, :, :
        ]  # [N, 2] in normalized cropped coords [-0.5, 0.5]

        # Convert GT keypoints from normalized cropped coords to full image coordinates
        affine_trans = batch["affine_trans"][batch_idx, 0]  # [2, 3] or [3, 3]
        img_size = batch["img_size"][batch_idx, 0]  # [2] (width, height)

        # Denormalize using img_size: (normalized + 0.5) * img_size
        # This gives cropped pixel coordinates
        gt_kp2d_denormalized = (gt_kp2d_normalized + 0.5) * img_size.reshape(
            1, 2
        )  # [N, 2]

        # Convert to homogeneous coordinates and apply inverse affine transformation
        gt_kp2d_homogeneous = np.ones((gt_kp2d_normalized.shape[0], 3))
        gt_kp2d_homogeneous[:, :2] = gt_kp2d_denormalized

        # Inverse affine transformation: need to compute inverse of affine_trans
        if affine_trans.shape == (2, 3):
            # For 2x3 matrix, we need to augment it to 3x3 for inversion
            affine_3x3 = np.eye(3)
            affine_3x3[:2, :] = affine_trans
            affine_inv = np.linalg.inv(affine_3x3)
            gt_kp2d_transformed = gt_kp2d_homogeneous @ affine_inv.T
            gt_kp2d_full = gt_kp2d_transformed[:, :2]
        elif affine_trans.shape == (3, 3):
            affine_inv = np.linalg.inv(affine_trans)
            gt_kp2d_transformed = gt_kp2d_homogeneous @ affine_inv.T
            gt_kp2d_full = gt_kp2d_transformed[:, :2]
        else:
            # Fallback: assume no transformation needed
            gt_kp2d_full = gt_kp2d_denormalized

        # Extract sample keypoints (already in full image coords)
        sample_kp2d_full = predictions["kp2d_samples"][
            batch_idx
        ]  # [num_samples, 70, 2]
        num_samples = sample_kp2d_full.shape[0]
        if self.max_plots is not None:
            num_samples = min(num_samples, self.max_plots)

        # Create visualization
        plt.figure(figsize=(15, 10))
        plt.imshow(image_original)

        # Plot GT keypoints
        plt.scatter(
            gt_kp2d_full[:, 0],
            gt_kp2d_full[:, 1],
            color="lime",
            s=10,
            marker="x",
            label="GT",
            linewidths=1,
        )

        # Plot predicted mean keypoints
        plt.scatter(
            pred_kp2d_full[:, 0],
            pred_kp2d_full[:, 1],
            color="red",
            s=10,
            marker="x",
            label="Pred Mean",
            linewidths=1,
        )

        # Plot sample keypoints - use different blues
        # Plot sample keypoints
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
        for i in range(num_samples):
            plt.scatter(
                sample_kp2d_full[i, :, 0],
                sample_kp2d_full[i, :, 1],
                color=colors[i],
                s=10,
                marker=".",
                alpha=0.6,
                label=f"Sample {i+1}" if i < 5 else None,
            )  # Only label first 5

        plt.legend()
        plt.title(f"2D Keypoints Visualization on Full Image (Batch {batch_idx})")
        plt.tight_layout()

        # Save using the visualiser's filename convention
        filename = self._get_filename("_keypoints_2d_full")
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualise_2d_keypoints_cropped(self, predictions, batch):
        batch_idx = 0
        img_size = batch["img_size"][batch_idx, 0][0]

        gt_kp2d_normalized = batch["keypoints_2d"][batch_idx, :70, :]  # joints_2d
        gt_kp2d = (gt_kp2d_normalized + 0.5) * img_size  # [N, 2]

        # Visibility mask for the first 70 keypoints (1 = visible, 0 = invisible)
        visibility = batch["visibility"][batch_idx]  # [N]
        visible_mask = visibility
        invisible_mask = ~visible_mask

        pred_kp2d_cropped_normalised = predictions["mhr"]["pred_keypoints_2d_cropped"][
            batch_idx
        ]  # [N, 2]
        pred_kp2d_cropped_coords = (
            pred_kp2d_cropped_normalised + 0.5
        ) * img_size  # [N, 2]

        sample_kp2d_cropped_normalized = predictions[
            "kp2d_samples_cropped"
        ][  # j2d_samples_cropped
            batch_idx
        ]  # [num_samples, N, 2]
        num_samples = sample_kp2d_cropped_normalized.shape[0]
        if self.max_plots is not None:
            num_samples = min(num_samples, self.max_plots)
        # Unnormalize to pixel coordinates [0, 256]
        sample_kp2d_cropped_coords = (
            sample_kp2d_cropped_normalized + 0.5
        ) * img_size  # [num_samples, N, 2]

        # Get cropped image
        img = batch["img"][batch_idx, 0]  # [3, 256, 256] or [256, 256, 3]
        img = ((img.transpose(1, 2, 0)) * 255).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        # Plot GT keypoints: visible vs invisible use similar but distinct symbols
        if visible_mask.any():
            plt.scatter(
                gt_kp2d[visible_mask, 0],
                gt_kp2d[visible_mask, 1],
                color="lime",
                s=10,
                marker="x",
                label="GT (visible)",
                linewidths=1,
            )
        if invisible_mask.any():
            plt.scatter(
                gt_kp2d[invisible_mask, 0],
                gt_kp2d[invisible_mask, 1],
                facecolors="none",
                edgecolors="lime",
                s=30,
                marker="o",
                label="GT (invisible)",
                linewidths=1,
            )

        # Plot predicted mean keypoints: use GT visibility mask for consistency
        if visible_mask.any():
            plt.scatter(
                pred_kp2d_cropped_coords[visible_mask, 0],
                pred_kp2d_cropped_coords[visible_mask, 1],
                color="red",
                s=10,
                marker="x",
                label="Pred Mean (visible)",
                linewidths=1,
            )
        if invisible_mask.any():
            plt.scatter(
                pred_kp2d_cropped_coords[invisible_mask, 0],
                pred_kp2d_cropped_coords[invisible_mask, 1],
                facecolors="none",
                edgecolors="red",
                s=30,
                marker="o",
                label="Pred Mean (invisible)",
                linewidths=1,
            )

        # Plot sample keypoints if available
        if sample_kp2d_cropped_coords is not None:
            colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
            for i in range(num_samples):
                # Visible sample keypoints
                if visible_mask.any():
                    plt.scatter(
                        sample_kp2d_cropped_coords[i, visible_mask, 0],
                        sample_kp2d_cropped_coords[i, visible_mask, 1],
                        color=colors[i],
                        s=8,
                        marker=".",
                        alpha=0.6,
                        label=f"Sample {i+1}" if i < 5 else None,
                    )  # Only label first 5
                # Invisible sample keypoints: similar but different symbol (hollow circles)
                if invisible_mask.any():
                    plt.scatter(
                        sample_kp2d_cropped_coords[i, invisible_mask, 0],
                        sample_kp2d_cropped_coords[i, invisible_mask, 1],
                        facecolors="none",
                        edgecolors=colors[i],
                        s=20,
                        marker="o",
                        alpha=0.6,
                    )

        plt.legend()
        plt.title(f"2D Keypoints Visualization (Batch {batch_idx})")
        plt.tight_layout()

        # Save using the visualiser's filename convention
        filename = self._get_filename("_keypoints_2d_cropped")
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def ori_visualise_2d_keypoints_cropped(self, predictions, batch):
        """
        Visualize ground truth, predicted mean, and sampled 2D keypoints on the cropped image.

        Args:
            predictions: Model predictions dictionary (already converted to numpy)
            batch: Input batch dictionary (already converted to numpy)
        """
        if "kp2d_samples" not in predictions:
            logger.warning(
                "No sample keypoints found in predictions. Skipping 2D keypoint visualization."
            )
            return

        # For consistency with other visualisation functions, only visualise the first element in the batch
        batch_idx = 0

        img_size = batch["img_size"][batch_idx, 0][0]

        gt_kp2d_normalized = batch["keypoints_2d"][
            batch_idx, :, :
        ]  # [N, 2] in normalized coords [-0.5, 0.5]
        gt_kp2d = (gt_kp2d_normalized + 0.5) * img_size  # [N, 2]

        # Predicted keypoints in cropped normalized coords [-0.5, 0.5]
        pred_kp2d_cropped_normalised = predictions["mhr"]["pred_keypoints_2d_cropped"][
            batch_idx
        ]  # [N, 2]
        pred_kp2d_cropped_coords = (
            pred_kp2d_cropped_normalised + 0.5
        ) * img_size  # [N, 2]

        sample_kp2d_cropped_normalized = predictions["kp2d_samples_cropped"][
            batch_idx
        ]  # [num_samples, N, 2]
        num_samples = sample_kp2d_cropped_normalized.shape[0]
        if self.max_plots is not None:
            num_samples = min(num_samples, self.max_plots)
        # Unnormalize to pixel coordinates [0, 256]
        sample_kp2d_cropped_coords = (
            sample_kp2d_cropped_normalized + 0.5
        ) * img_size  # [num_samples, N, 2]

        # Get cropped image
        img = batch["img"][batch_idx, 0]  # [3, 256, 256] or [256, 256, 3]
        img = ((img.transpose(1, 2, 0)) * 255).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        # Plot GT keypoints
        plt.scatter(
            gt_kp2d[:, 0],
            gt_kp2d[:, 1],
            color="lime",
            s=10,
            marker="x",
            label="GT",
            linewidths=1,
        )

        # Plot predicted mean keypoints
        plt.scatter(
            pred_kp2d_cropped_coords[:, 0],
            pred_kp2d_cropped_coords[:, 1],
            color="red",
            s=10,
            marker="x",
            label="Pred Mean",
            linewidths=1,
        )

        # Plot sample keypoints if available
        if sample_kp2d_cropped_coords is not None:
            colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
            for i in range(num_samples):
                plt.scatter(
                    sample_kp2d_cropped_coords[i, :, 0],
                    sample_kp2d_cropped_coords[i, :, 1],
                    color=colors[i],
                    s=8,
                    marker=".",
                    alpha=0.6,
                    label=f"Sample {i+1}" if i < 5 else None,
                )  # Only label first 5

        plt.legend()
        plt.title(f"2D Keypoints Visualization (Batch {batch_idx})")
        plt.tight_layout()

        # Save using the visualiser's filename convention
        filename = self._get_filename("_keypoints_2d_cropped")
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualise_merging(
        self, predictions, batch=None, save_path=None, suffix=None, normalise=True
    ):
        """
        visualise the ground truth mesh, the predicted mesh, and the merged mesh using 3D scatter plots
        Shows per-view visualizations: GT, predicted, and merged for each view

        Args:
            predictions: dict containing gt_vertices, pred_vertices, mu_star_vertices (or neutral versions)
            batch: optional batch data
            save_path: optional custom save path
            suffix: optional suffix to add to default save path (e.g., 'neutral')
            normalise: if True, normalize height of meshes wrt GT before error calculation and visualization (default: True)
        """
        # Extract data from predictions dict
        # predictions should have: gt_vertices, pred_vertices, mu_star_vertices
        # All should be [N_views, N_verts, 3] or [B, N_views, N_verts, 3]

        # Handle batch dimension
        gt_verts_all_views = predictions[
            "gt_vertices"
        ]  # [N_views, N_verts, 3] or [B, N_views, N_verts, 3]

        pred_verts_all_views = predictions[
            "pred_vertices"
        ]  # [N_views, N_verts, 3] or [B, N_views, N_verts, 3]
        merged_verts_all_views = predictions[
            "mu_star_vertices"
        ]  # [N_views, N_verts, 3] or [B, N_views, N_verts, 3]
        print(
            gt_verts_all_views.shape,
            pred_verts_all_views.shape,
            merged_verts_all_views.shape,
        )

        # Apply camera coordinate system transformation to all vertices for consistent visualization
        gt_verts_all_views[..., [1, 2]] *= -1
        pred_verts_all_views[..., [1, 2]] *= -1
        merged_verts_all_views[..., [1, 2]] *= -1

        # Convert to numpy and handle batch dimension
        if isinstance(pred_verts_all_views, torch.Tensor):
            pred_verts_all_views = pred_verts_all_views.detach().cpu().numpy()
        if isinstance(merged_verts_all_views, torch.Tensor):
            merged_verts_all_views = merged_verts_all_views.detach().cpu().numpy()
        if isinstance(gt_verts_all_views, torch.Tensor):
            gt_verts_all_views = gt_verts_all_views.detach().cpu().numpy()

        # Remove batch dimension if present
        if pred_verts_all_views.ndim == 4:
            pred_verts_all_views = pred_verts_all_views[0]  # [N_views, N_verts, 3]
        if merged_verts_all_views.ndim == 4:
            merged_verts_all_views = merged_verts_all_views[0]  # [N_views, N_verts, 3]
        if gt_verts_all_views.ndim == 4:
            gt_verts_all_views = gt_verts_all_views[0]  # [N_views, N_verts, 3]

        num_views = pred_verts_all_views.shape[0]

        # For neutral pose, apply scale and translation correction (like PVETSC)
        if suffix == "neutral":
            from sam_3d_body.metrics.metrics_tracker import (
                scale_and_translation_transform_batch,
            )

            # Apply scale and translation correction to predicted and merged vertices
            for view_idx in range(num_views):
                gt_verts = gt_verts_all_views[view_idx]  # [N_verts, 3]
                pred_verts = pred_verts_all_views[view_idx]  # [N_verts, 3]
                merged_verts = merged_verts_all_views[view_idx]  # [N_verts, 3]

                # Apply scale and translation correction: normalize pred/merged to match GT scale and translation
                pred_verts_corrected = scale_and_translation_transform_batch(
                    pred_verts[np.newaxis, :, :],  # [1, N_verts, 3]
                    gt_verts[np.newaxis, :, :],  # [1, N_verts, 3]
                )[
                    0
                ]  # [N_verts, 3]

                merged_verts_corrected = scale_and_translation_transform_batch(
                    merged_verts[np.newaxis, :, :],  # [1, N_verts, 3]
                    gt_verts[np.newaxis, :, :],  # [1, N_verts, 3]
                )[
                    0
                ]  # [N_verts, 3]

                # Replace with corrected vertices
                pred_verts_all_views[view_idx] = pred_verts_corrected
                merged_verts_all_views[view_idx] = merged_verts_corrected

        # Normalize height of meshes relative to GT if requested
        if normalise:
            for view_idx in range(num_views):
                gt_verts = gt_verts_all_views[view_idx]  # [N_verts, 3]
                pred_verts = pred_verts_all_views[view_idx]  # [N_verts, 3]
                merged_verts = merged_verts_all_views[view_idx]  # [N_verts, 3]

                # Calculate height (Y-axis extent) for each mesh
                gt_height = np.max(gt_verts[:, 1]) - np.min(gt_verts[:, 1])
                pred_height = np.max(pred_verts[:, 1]) - np.min(pred_verts[:, 1])
                merged_height = np.max(merged_verts[:, 1]) - np.min(merged_verts[:, 1])

                # Normalize predicted and merged meshes to match GT height
                if pred_height > 1e-6:
                    pred_scale = gt_height / pred_height
                    pred_verts_all_views[view_idx] = pred_verts * pred_scale

                if merged_height > 1e-6:
                    merged_scale = gt_height / merged_height
                    merged_verts_all_views[view_idx] = merged_verts * merged_scale

        # Calculate distances for all views first to determine global colormap range
        all_pred_distances = []
        all_merged_distances = []

        for view_idx in range(num_views):
            pred_verts = pred_verts_all_views[view_idx]  # [N_verts, 3]
            merged_verts = merged_verts_all_views[view_idx]  # [N_verts, 3]
            gt_verts = gt_verts_all_views[view_idx]  # [N_verts, 3]

            # Calculate distances from predicted and merged vertices to GT
            pred_distances = np.linalg.norm(pred_verts - gt_verts, axis=1)  # [N_verts]
            merged_distances = np.linalg.norm(
                merged_verts - gt_verts, axis=1
            )  # [N_verts]

            all_pred_distances.append(pred_distances)
            all_merged_distances.append(merged_distances)

        # Find global maximum distance across all views and all prediction types
        global_max_distance = max(
            max([d.max() for d in all_pred_distances]) if all_pred_distances else 0,
            max([d.max() for d in all_merged_distances]) if all_merged_distances else 0,
        )

        # Convert to centimeters for display
        global_max_distance_cm = global_max_distance * 100

        # Create figure with num_views rows and 3 columns (GT, Predicted, Merged)
        fig = plt.figure(figsize=(18, 6 * num_views))

        for view_idx in range(num_views):
            # Get vertices for this view
            pred_verts = pred_verts_all_views[view_idx]  # [N_verts, 3]
            merged_verts = merged_verts_all_views[view_idx]  # [N_verts, 3]
            gt_verts = gt_verts_all_views[view_idx]  # [N_verts, 3]

            # Get pre-computed distances for this view
            pred_distances = all_pred_distances[view_idx]  # [N_verts] in meters
            merged_distances = all_merged_distances[view_idx]  # [N_verts] in meters

            # Calculate average distances in centimeters
            avg_pred_distance_cm = pred_distances.mean() * 100
            avg_merged_distance_cm = merged_distances.mean() * 100

            # Convert distances to centimeters for colormap (use actual distances, not normalized)
            pred_distances_cm = pred_distances * 100  # [N_verts] in cm
            merged_distances_cm = merged_distances * 100  # [N_verts] in cm

            # Convert global max to centimeters for consistent colormap scale
            global_max_distance_cm = global_max_distance * 100

            # Ground truth vertices
            ax1 = fig.add_subplot(num_views, 3, view_idx * 3 + 1, projection="3d")
            ax1.scatter(
                gt_verts[:, 0],
                gt_verts[:, 1],
                gt_verts[:, 2],
                c="blue",
                s=1,
                alpha=0.6,
                label="GT Vertices",
            )
            ax1.set_title(f"View {view_idx+1}: Ground Truth Vertices")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            # Predicted vertices for this view (colored by distance to GT)
            ax2 = fig.add_subplot(num_views, 3, view_idx * 3 + 2, projection="3d")
            scatter2 = ax2.scatter(
                pred_verts[:, 0],
                pred_verts[:, 1],
                pred_verts[:, 2],
                c=pred_distances_cm,
                s=1,
                alpha=0.6,
                cmap="viridis",
                vmin=0,
                vmax=global_max_distance_cm,
                label="Predicted Vertices",
            )
            ax2.set_title(
                f"View {view_idx+1}: Predicted Vertices\n(Avg distance: {avg_pred_distance_cm:.2f} cm)"
            )
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Distance to GT (cm)", rotation=270, labelpad=15)

            # Merged vertices for this view (colored by distance to GT)
            ax3 = fig.add_subplot(num_views, 3, view_idx * 3 + 3, projection="3d")
            scatter3 = ax3.scatter(
                merged_verts[:, 0],
                merged_verts[:, 1],
                merged_verts[:, 2],
                c=merged_distances_cm,
                s=1,
                alpha=0.6,
                cmap="viridis",
                vmin=0,
                vmax=global_max_distance_cm,
                label="Merged Vertices",
            )
            ax3.set_title(
                f"View {view_idx+1}: Merged Vertices\n(Avg distance: {avg_merged_distance_cm:.2f} cm)"
            )
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")
            cbar3 = plt.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label("Distance to GT (cm)", rotation=270, labelpad=15)

            # Set equal aspect ratio for all subplots in this row
            all_verts = np.concatenate([gt_verts, pred_verts, merged_verts], axis=0)

            max_range = (
                np.array(
                    [
                        all_verts[:, 0].max() - all_verts[:, 0].min(),
                        all_verts[:, 1].max() - all_verts[:, 1].min(),
                        all_verts[:, 2].max() - all_verts[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )
            mid_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) * 0.5
            mid_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) * 0.5
            mid_z = (all_verts[:, 2].max() + all_verts[:, 2].min()) * 0.5

            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.view_init(elev=10, azim=20, vertical_axis="y")

        plt.tight_layout()

        # Save if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved merging visualization to {save_path}")
        else:
            # Construct default save path with optional suffix
            if suffix is not None:
                filename = f"merging_visualization_{suffix}.png"
            else:
                filename = "merging_visualization.png"
            default_path = os.path.join(self.save_dir, filename)
            plt.savefig(default_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved merging visualization to {default_path}")

        plt.close(fig)



################################################################################
# Multiview visualisations
################################################################################

def vis_merging_predictions(
    input_dict,
    save_dir: str = None,
    plot_side: bool = True,
    overlay_sideview: bool = True,
    sc: bool = True,
):
    num_views = input_dict["num_views"]
    bs = input_dict["bs"]
    batch_idx = input_dict["batch_idx"]

    renderer = input_dict["renderer"]
    side_renderer = input_dict["neutral_renderer"]

    outputs = input_dict["outputs"]
    batch = input_dict["batch"]
    # merged_verts = input_dict.get("verts_star", None)
    merged_verts = input_dict.get("merged_verts", None)
    merged_neutral_verts = input_dict.get("merged_neutral_verts", None)

    metrics = input_dict.get("metrics", {})

    pred_shape = input_dict.get("pred_shape", None)
    merged_shape = input_dict.get("shape_mu_star", None)
    shape_var_unflattened = input_dict.get("shape_var_unflattened", None)
    merged_shape_var = input_dict.get("merged_shape_var", None)
    
    merged_neutral_verts = input_dict.get("merged_neutral_verts", None)
    has_merged = merged_verts is not None and merged_neutral_verts is not None

    # Initialize gallery: list of lists to store rendered images [bs][num_views]
    gallery = [[None for _ in range(num_views)] for _ in range(bs)]

    i = 0
    all_distances = []
    pred_vertex_dists = {}
    merged_vertex_dists = {}
    gt_centered_for_side = {}
    verts_centered_for_side = {}
    merged_centered_for_side = {}

    for view in range(num_views):

        flat_idx = i * num_views + view
        pred_verts = outputs["mhr"]["pred_vertices"][flat_idx].cpu().detach().numpy()
        gt_verts = batch["vertices"][flat_idx].cpu().detach().numpy()
        if not has_merged:
            # `merged_verts` and `merged_neutral_verts` should both be provided for this visualization.
            assert False
        merged_verts_view = merged_verts[flat_idx].cpu().detach().numpy()

        if sc:
            # PVETS-T-C: scale + translation normalize predicted vertices to GT.
            pred_verts = scale_and_translation_transform_batch(
                pred_verts[None, ...], gt_verts[None, ...]
            )[0]
            merged_verts_view = scale_and_translation_transform_batch(
                merged_verts_view[None, ...], gt_verts[None, ...]
            )[0]

        # Pre-compute centered coordinates for the side-view renderer.
        # These match the centering logic that used to happen inside `plot_side`.
        center = gt_verts.mean(axis=0, keepdims=True)
        gt_centered_for_side[view] = gt_verts - center
        verts_centered_for_side[view] = pred_verts - center
        merged_centered_for_side[view] = merged_verts_view - center

        # Distances are in meters; convert to millimeters for visualization.
        pred_dist = np.linalg.norm(pred_verts - gt_verts, axis=1) * 1000.0
        merged_dist = np.linalg.norm(merged_verts_view - gt_verts, axis=1) * 1000.0

        pred_vertex_dists[view] = pred_dist
        merged_vertex_dists[view] = merged_dist

        all_distances.append(pred_dist)
        all_distances.append(merged_dist)

    all_distances = np.concatenate(all_distances)
    min_dist = float(all_distances.min()) if all_distances.size > 0 else 0.0
    max_dist = float(all_distances.max()) if all_distances.size > 0 else 1.0

    # Second pass: render GT (solid color), and pred/merged with per-vertex viridis colors
    for view in range(num_views):
        img_for_render = batch["img_ori"][view][i].cpu().detach().numpy()

        flat_idx = i * num_views + view

        verts = outputs["mhr"]["pred_vertices"][flat_idx].cpu().detach().numpy()
        cam_t = outputs["mhr"]["pred_cam_t"][flat_idx].cpu().detach().numpy()

        gt_verts = batch["vertices"][flat_idx].cpu().detach().numpy()
        # gt_verts[..., [1, 2]] *= -1  # un-flip for renderer (renderer applies 180-deg X internally)
        if "cam_ext" not in batch:
            # SSP-3D
            assert batch["dataset_name"][0] == "ssp3d"
            gt_cam_t = batch["trans_cam"][flat_idx].cpu().detach().numpy()
        else:
            gt_cam_t = batch["cam_ext"][flat_idx][:3, -1].cpu().detach().numpy()

        if has_merged:
            merged_verts_view = merged_verts[flat_idx].cpu().detach().numpy()
        else:
            assert False

        # GT: keep fixed LIGHT_BLUE color
        gt_rendered_img = (
            renderer(
                gt_verts,
                gt_cam_t,
                img_for_render.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                camera_center=(
                    batch["cam_int"][flat_idx][0, 2],
                    batch["cam_int"][flat_idx][1, 2],
                ),
            )
            * 255
        ).astype(np.uint8)

        # Predicted mesh: per-vertex viridis colors from distance to GT
        pred_colors = build_vertex_colors(
            pred_vertex_dists[view], min_dist=min_dist, max_dist=max_dist
        )
        rendered_img = (
            renderer(
                verts,
                cam_t,
                img_for_render.copy(),
                mesh_base_color=(1.0, 0.8, 0.5),
                scene_bg_color=(1, 1, 1),
                camera_center=(
                    batch["cam_int"][flat_idx][0, 2],
                    batch["cam_int"][flat_idx][1, 2],
                ),
                vertex_colors=pred_colors,
            )
            * 255
        ).astype(np.uint8)

        # Overlay semi-transparent GT mesh on top of predicted mesh (light orange)
        gt_rgba = renderer(
            gt_verts,
            gt_cam_t,
            np.ones_like(img_for_render) * 255,
            mesh_base_color=(1.0, 0.8, 0.5),
            scene_bg_color=(1, 1, 1),
            camera_center=(
                batch["cam_int"][flat_idx][0, 2],
                batch["cam_int"][flat_idx][1, 2],
            ),
            return_rgba=True,
        )
        alpha = gt_rgba[..., 3:4].astype(np.float32) * 0.5
        pred_rgb = rendered_img.astype(np.float32) / 255.0
        gt_rgb = gt_rgba[..., :3].astype(np.float32)
        blended_pred = alpha * gt_rgb + (1.0 - alpha) * pred_rgb
        rendered_img = (blended_pred * 255.0).clip(0, 255).astype(np.uint8)

        # Merged mesh: per-vertex viridis colors from distance to GT
        merged_colors = build_vertex_colors(
            merged_vertex_dists[view], min_dist=min_dist, max_dist=max_dist
        )
        rendered_merged_img = (
            renderer(
                merged_verts_view,
                cam_t,
                img_for_render.copy(),
                mesh_base_color=(0.5, 1.0, 0.5),
                scene_bg_color=(1, 1, 1),
                camera_center=(
                    batch["cam_int"][flat_idx][0, 2],
                    batch["cam_int"][flat_idx][1, 2],
                ),
                vertex_colors=merged_colors,
            )
            * 255
        ).astype(np.uint8)

        # Overlay semi-transparent GT mesh on top of merged mesh (light orange)
        gt_rgba_merged = renderer(
            gt_verts,
            gt_cam_t,
            np.ones_like(img_for_render) * 255,
            mesh_base_color=(1.0, 0.8, 0.5),
            scene_bg_color=(1, 1, 1),
            camera_center=(
                batch["cam_int"][flat_idx][0, 2],
                batch["cam_int"][flat_idx][1, 2],
            ),
            return_rgba=True,
        )
        alpha_m = gt_rgba_merged[..., 3:4].astype(np.float32) * 0.3
        merged_rgb = rendered_merged_img.astype(np.float32) / 255.0
        gt_m_rgb = gt_rgba_merged[..., :3].astype(np.float32)
        blended_merged = alpha_m * gt_m_rgb + (1.0 - alpha_m) * merged_rgb
        rendered_merged_img = (blended_merged * 255.0).clip(0, 255).astype(np.uint8)

        affine = batch["affine_trans"][flat_idx, 0].cpu().detach().numpy()
        img_size = batch["img_size"][flat_idx, 0].cpu().detach().numpy()

        gt_rendered_img = cv2.warpAffine(gt_rendered_img, affine, img_size)
        rendered_img = cv2.warpAffine(rendered_img, affine, img_size)
        rendered_merged_img = cv2.warpAffine(rendered_merged_img, affine, img_size)

        # Add text labels to images
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_scale_small = 0.6
        thickness = 2
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background for text

        # Label gt image
        gt_label = f"GT {view}"
        (text_width, text_height), baseline = cv2.getTextSize(
            gt_label, font, font_scale, thickness
        )
        cv2.rectangle(
            gt_rendered_img,
            (10, 10),
            (10 + text_width + 4, 10 + text_height + baseline + 4),
            bg_color,
            -1,
        )
        cv2.putText(
            gt_rendered_img,
            gt_label,
            (12, 10 + text_height),
            font,
            font_scale,
            color,
            thickness,
        )

        # Label GT image
        gt_label = f"GT view {view}"
        (text_width, text_height), baseline = cv2.getTextSize(
            gt_label, font, font_scale, thickness
        )
        text_lines = []
        has_pampjpe = "per_view_pampjpe" in metrics and "merged_pampjpe" in metrics and len(metrics["per_view_pampjpe"]) > 0 and len(metrics["merged_pampjpe"]) > 0
        has_pvetsc = "per_view_pvetsc" in metrics and "merged_pvetsc" in metrics and len(metrics["per_view_pvetsc"]) > 0 and len(metrics["merged_pvetsc"]) > 0
        if has_pampjpe:
            per_view_pampjpe = metrics["per_view_pampjpe"][-1]
            merged_pampjpe = metrics["merged_pampjpe"][-1]
            text_lines.append(
                f"PA-MPJPE | View {view}: {per_view_pampjpe[view].item():.4f} | Merged: {merged_pampjpe.mean().item():.4f}"
            )
        if has_pvetsc:
            per_view_pvetsc = metrics["per_view_pvetsc"][-1]
            merged_pvetsc = metrics["merged_pvetsc"][-1]
            text_lines.append(
                f"PVE-T-SC | View {view}: {per_view_pvetsc[view].item():.4f} | Merged: {merged_pvetsc.mean().item():.4f}"
            )

        y_start = 10 + text_height + baseline + 10
        y_offset = y_start
        for line in text_lines:
            (tw, th), bl = cv2.getTextSize(line, font, font_scale_small, thickness)
            cv2.rectangle(
                gt_rendered_img,
                (10, y_offset),
                (10 + tw + 4, y_offset + th + bl + 4),
                bg_color,
                -1,
            )
            cv2.putText(
                gt_rendered_img,
                line,
                (12, y_offset + th),
                font,
                font_scale_small,
                color,
                thickness,
            )
            y_offset += th + bl + 6

        # Label predicted image
        pred_label = f"Pred view {view}"
        (text_width, text_height), baseline = cv2.getTextSize(
            pred_label, font, font_scale, thickness
        )
        cv2.rectangle(
            rendered_img,
            (10, 10),
            (10 + text_width + 4, 10 + text_height + baseline + 4),
            bg_color,
            -1,
        )
        cv2.putText(
            rendered_img,
            pred_label,
            (12, 10 + text_height),
            font,
            font_scale,
            color,
            thickness,
        )

        # Add per-view predicted shape parameters (first 5) and uncertainties
        pred_text_lines = []
        if pred_shape is not None:
            pred_mu = pred_shape[i, view].cpu().detach().numpy()
            pred_text_lines.append(
                "pred shape mean: " + " ".join(f"{v:.2f}" for v in pred_mu[:5])
            )
        if shape_var_unflattened is not None:
            pred_var = shape_var_unflattened[i, view].cpu().detach().numpy()
            pred_text_lines.append(
                "pred shape var: " + " ".join(f"{v:.2f}" for v in pred_var[:5])
            )
        y_start = 10 + text_height + baseline + 10
        y_offset = y_start
        for line in pred_text_lines:
            (tw, th), bl = cv2.getTextSize(line, font, font_scale_small, thickness)
            cv2.rectangle(
                rendered_img,
                (10, y_offset),
                (10 + tw + 4, y_offset + th + bl + 4),
                bg_color,
                -1,
            )
            cv2.putText(
                rendered_img,
                line,
                (12, y_offset + th),
                font,
                font_scale_small,
                color,
                thickness,
            )
            y_offset += th + bl + 6

        # Label merged image
        merged_label = f"Merged view {view}"
        (text_width, text_height), baseline = cv2.getTextSize(
            merged_label, font, font_scale, thickness
        )
        cv2.rectangle(
            rendered_merged_img,
            (10, 10),
            (10 + text_width + 4, 10 + text_height + baseline + 4),
            bg_color,
            -1,
        )
        cv2.putText(
            rendered_merged_img,
            merged_label,
            (12, 10 + text_height),
            font,
            font_scale,
            color,
            thickness,
        )

        # Add merged shape parameters (first 5) and uncertainties
        merged_text_lines = []
        if merged_shape is not None:
            merged_mu = merged_shape[i].cpu().detach().numpy()
            merged_text_lines.append(
                "merged shape mean: " + " ".join(f"{v:.2f}" for v in merged_mu[:5])
            )
        if merged_shape_var is not None:
            merged_var = merged_shape_var[i].cpu().detach().numpy()
            merged_text_lines.append(
                "merged shape var: " + " ".join(f"{v:.2f}" for v in merged_var[:5])
            )
        if merged_verts is None:
            merged_text_lines.append("merged prediction unavailable")
        y_start_m = 10 + text_height + baseline + 10
        y_offset_m = y_start_m
        for line in merged_text_lines:
            (tw, th), bl = cv2.getTextSize(line, font, font_scale_small, thickness)
            cv2.rectangle(
                rendered_merged_img,
                (10, y_offset_m),
                (10 + tw + 4, y_offset_m + th + bl + 4),
                bg_color,
                -1,
            )
            cv2.putText(
                rendered_merged_img,
                line,
                (12, y_offset_m + th),
                font,
                font_scale_small,
                color,
                thickness,
            )
            y_offset_m += th + bl + 6

        if plot_side:
            white_bg = np.ones_like(gt_rendered_img) #np.ones_like(img_for_render) * 255
            generic_cam_t = np.array([0.0, -0.25, 2.5])
            gt_centered = gt_centered_for_side[view]
            verts_centered = verts_centered_for_side[view]
            merged_centered = merged_centered_for_side[view]

            # GT side view (for the GT column)
            gt_side = (
                side_renderer(
                    gt_centered,
                    generic_cam_t,
                    white_bg.copy(),
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    side_view=True,
                    rot_angle=90,
                )
                * 255
            ).astype(np.uint8)

            # Pred side view with GT overlay
            pred_side_base = (
                side_renderer(
                    verts_centered,
                    generic_cam_t,
                    white_bg.copy(),
                    mesh_base_color=(1.0, 0.8, 0.5),
                    scene_bg_color=(1, 1, 1),
                    vertex_colors=pred_colors,
                    side_view=True,
                    rot_angle=90,
                )
                * 255
            ).astype(np.uint8)
            gt_side_rgba = side_renderer(
                gt_centered,
                generic_cam_t,
                white_bg.copy(),
                mesh_base_color=(1.0, 0.8, 0.5),
                scene_bg_color=(1, 1, 1),
                side_view=True,
                rot_angle=90,
                return_rgba=True,
            )
            alpha_side = gt_side_rgba[..., 3:4].astype(np.float32) * 0.5
            pred_side_rgb = pred_side_base.astype(np.float32) / 255.0
            gt_side_rgb = gt_side_rgba[..., :3].astype(np.float32)
            blended_pred_side = (
                alpha_side * gt_side_rgb + (1.0 - alpha_side) * pred_side_rgb
            )
            pred_side = (
                (blended_pred_side * 255.0).clip(0, 255).astype(np.uint8)
            )

            # Merged side view with GT overlay
            merged_side_base = (
                side_renderer(
                    merged_centered,
                    generic_cam_t,
                    white_bg.copy(),
                    mesh_base_color=(0.5, 1.0, 0.5),
                    scene_bg_color=(1, 1, 1),
                    vertex_colors=merged_colors,
                    side_view=True,
                    rot_angle=90,
                )
                * 255
            ).astype(np.uint8)
            merged_side_rgb = merged_side_base.astype(np.float32) / 255.0
            blended_merged_side = (
                alpha_side * gt_side_rgb + (1.0 - alpha_side) * merged_side_rgb
            )
            merged_side = (
                (blended_merged_side * 255.0).clip(0, 255).astype(np.uint8)
            )

            # gt_side = cv2.warpAffine(gt_side, affine, img_size)
            # pred_side = cv2.warpAffine(pred_side, affine, img_size)
            # merged_side = cv2.warpAffine(merged_side, affine, img_size)


            gallery[i][view] = np.concatenate(
                [
                    gt_rendered_img,
                    gt_side,
                    rendered_img,
                    pred_side,
                    rendered_merged_img,
                    merged_side,
                ],
                axis=1,
            )
        else:
            gallery[i][view] = np.concatenate(
                [gt_rendered_img, rendered_img, rendered_merged_img], axis=1
            )

    gallery_rows = []
    for i in range(1):
        row = np.concatenate(
            [gallery[i][view] for view in range(num_views)], axis=0
        )
        gallery_rows.append(row)

    # gallery_img = np.concatenate(gallery_rows, axis=0)
    gallery_img = gallery_rows[0]
    gallery_img_bgr = cv2.cvtColor(gallery_img, cv2.COLOR_RGB2BGR)
    # Downscale final image by factor 2 before saving
    h, w = gallery_img_bgr.shape[:2]
    gallery_img_bgr = cv2.resize(
        gallery_img_bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA
    )

    # Append error-distance colorbar to the right.
    colorbar_rgb = build_distance_colorbar_rgb(
        min_dist=min_dist,
        max_dist=max_dist,
        cmap="inferno",
        height=gallery_img_bgr.shape[0],
        width=60,
    )
    colorbar_bgr = cv2.cvtColor(colorbar_rgb, cv2.COLOR_RGB2BGR)
    gallery_img_bgr = np.concatenate([gallery_img_bgr, colorbar_bgr], axis=1)

    # save_dir = self.vis_save_dir if self.vis_save_dir else "."
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_sc" if sc else ""
    save_path = os.path.join(
        save_dir,
        # f"batch{batch_idx:03d}_bs{bs}_views{num_views}{suffix}.png",
        f"b{batch_idx:03d}{suffix}.png",
    )
    cv2.imwrite(save_path, gallery_img_bgr)
    logger.info(
        f"Saved multiview gallery: {save_path} (shape: {gallery_img.shape})"
    )

def vis_merging_neutral(
    input_dict,
    save_dir: str = None,
    sc: bool = True,
    plot_hist: bool = True,
    use_best_by_log_prob: bool = True,
):
    generic_cam_t = np.array([0.0, 0.75, 2.5])
    batch_idx = input_dict["batch_idx"]
    num_views = input_dict["num_views"]

    renderer = input_dict["neutral_renderer"]
    
    gt_verts = input_dict["gt_neutral_verts"].cpu().detach().numpy()
    using_best_per_view = use_best_by_log_prob and (
        "best_logprob_sample_neutral_verts" in input_dict
    )
    per_view_verts_key = (
        "best_logprob_sample_neutral_verts"
        if using_best_per_view
        else "per_view_neutral_verts"
    )
    per_view_verts = input_dict[per_view_verts_key].cpu().detach().numpy()
    merged_verts = input_dict.get("merged_neutral_verts", None)
    if merged_verts is not None:
        merged_verts = merged_verts.cpu().detach().numpy()

    metrics = input_dict.get("metrics", {})
    per_view_pvetsc = None
    merged_pvetsc = None
    if (
        isinstance(metrics, dict)
        and "per_view_pvetsc" in metrics
        and len(metrics["per_view_pvetsc"]) > 0
    ):
        per_view_pvetsc = metrics["per_view_pvetsc"][-1]
    if (
        isinstance(metrics, dict)
        and "merged_pvetsc" in metrics
        and len(metrics["merged_pvetsc"]) > 0
    ):
        merged_pvetsc = metrics["merged_pvetsc"][-1]

    per_view_pvetsc_render = per_view_pvetsc

    if sc:
        per_view_verts = scale_and_translation_transform_batch(
            per_view_verts, gt_verts
        )
        if merged_verts is not None:
            merged_verts = scale_and_translation_transform_batch(
                merged_verts, gt_verts
            )

    # ----------------- Get colors -----------------
    gt_ref = gt_verts[-1]

    all_distances = []
    per_view_vertex_dists = {}

    for view in range(num_views):
        pv_verts = per_view_verts[view]
        # Distances are in meters; convert to millimeters for visualization.
        dist_pv = np.linalg.norm(pv_verts - gt_ref, axis=1) * 1000.0
        per_view_vertex_dists[view] = dist_pv
        all_distances.append(dist_pv)

    if using_best_per_view and per_view_pvetsc_render is not None:
        # PVETSC is mean L2 distance over vertices (meters). Our distances are in mm.
        per_view_pvetsc_render = np.array(
            [per_view_vertex_dists[v].mean() / 1000.0 for v in range(num_views)],
            dtype=np.float32,
        )

    if merged_verts is not None:
        merged_verts_ref = merged_verts[0]
        merged_vertex_dists = (
            np.linalg.norm(merged_verts_ref - gt_ref, axis=1) * 1000.0
        )
        all_distances.append(merged_vertex_dists)

    all_distances = np.concatenate(all_distances)
    min_dist = float(all_distances.min()) if all_distances.size > 0 else 0.0
    max_dist = float(all_distances.max()) if all_distances.size > 0 else 1.0

    # ----------------- GT -----------------
    background = np.zeros((512, 512, 3)) * 255

    gt_verts_vis = gt_verts.copy()
    gt_verts_vis[..., [1, 2]] *= -1
    
    gt_rgba = renderer(
        gt_verts_vis[-1],
        generic_cam_t,
        background.copy(),
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        return_rgba=True,
    )
    gt_alpha = gt_rgba[..., 3:4].astype(np.float32) * 0.5
    gt_rgb = gt_rgba[..., :3].astype(np.float32)
    gt_front = (gt_rgb * 255.0).clip(0, 255).astype(np.uint8)

    (text_width, text_height), baseline = cv2.getTextSize(
        "GT", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )

    text_config = {
        "org": (12, 10 + text_height),
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1.0,
        "color": (255, 255, 255),
        "thickness": 2,
    }
    cv2.putText(
        gt_front,
        "GT",
        **text_config,
    )


    gt_side_rgba = renderer(
        gt_verts_vis[-1],
        generic_cam_t,
        background.copy(),
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        side_view=True,
        rot_angle=90,
        return_rgba=True,
    )
    gt_side_alpha = gt_side_rgba[..., 3:4].astype(np.float32) * 0.5
    gt_side_rgb = gt_side_rgba[..., :3].astype(np.float32)
    gt_side = (gt_side_rgb * 255.0).clip(0, 255).astype(np.uint8)

    # ----------------- Per-view -----------------
    per_view_front = []
    per_view_side = []
    per_view_verts_vis = per_view_verts.copy()
    per_view_verts_vis[..., [1, 2]] *= -1

    for view in range(num_views):
        vertex_colors = build_vertex_colors(
            per_view_vertex_dists[view], min_dist=min_dist, max_dist=max_dist
        )
        front_render = renderer(
            per_view_verts_vis[view],
            generic_cam_t,
            background.copy(),
            mesh_base_color=(1.0, 0.8, 0.5), 
            scene_bg_color=(1, 1, 1),
            vertex_colors=vertex_colors,
        )
        pv_rgb = front_render.astype(np.float32)
        blended_pv = gt_alpha * gt_rgb + (1.0 - gt_alpha) * pv_rgb
        front_render = (blended_pv * 255.0).clip(0, 255).astype(np.uint8)
        per_view_front.append(front_render)

        cv2.putText(
            front_render,
            f"View {view}",
            **text_config,
        )

        side_render = renderer(
            per_view_verts_vis[view],
            generic_cam_t,
            background.copy(),
            mesh_base_color=(1.0, 0.8, 0.5),
            scene_bg_color=(1, 1, 1),
            vertex_colors=vertex_colors,
            side_view=True,
            rot_angle=90,
        )
        pv_side_rgb = side_render.astype(np.float32)
        blended_side = gt_side_alpha * gt_side_rgb + (1.0 - gt_side_alpha) * pv_side_rgb
        side_render = (blended_side * 255.0).clip(0, 255).astype(np.uint8)

        if per_view_pvetsc_render is not None:
            metric_config_side_view = {
                "org": (12, 10 + 2 * text_height + baseline),
                "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale": 0.75,
                "color": (255, 255, 255),
                "thickness": 2,
            }
            cv2.putText(
                side_render,
                f"PVE-T-SC: {float(per_view_pvetsc_render[view]) * 1000.0:.1f} mm",
                **metric_config_side_view,
            )
        per_view_side.append(side_render)

    # ----------------- Merged -----------------
    if merged_verts is not None:
        merged_verts_vis = merged_verts_ref.copy()
        merged_verts_vis[..., [1, 2]] *= -1
        merged_vertex_colors = build_vertex_colors(
            merged_vertex_dists, min_dist=min_dist, max_dist=max_dist
        )

        merged_front_render = renderer(
            merged_verts_vis,
            generic_cam_t,
            background.copy(),
            mesh_base_color=(0.5, 1.0, 0.5), 
            scene_bg_color=(1, 1, 1),
            vertex_colors=merged_vertex_colors,
        )
        merged_rgb = merged_front_render.astype(np.float32)
        blended_merged = gt_alpha * gt_rgb + (1.0 - gt_alpha) * merged_rgb
        merged_front_render = (blended_merged * 255.0).clip(0, 255).astype(np.uint8)

        cv2.putText(
            merged_front_render,
            "Merged",
            **text_config,
        )

        merged_side = renderer(
            merged_verts_vis,
            generic_cam_t,
            background.copy(),
            mesh_base_color=(0.5, 1.0, 0.5),
            scene_bg_color=(1, 1, 1),
            vertex_colors=merged_vertex_colors,
            side_view=True,
            rot_angle=90,
        )
        merged_side_rgb = merged_side.astype(np.float32)
        blended_merged_side = gt_side_alpha * gt_side_rgb + (1.0 - gt_side_alpha) * merged_side_rgb
        merged_side = (blended_merged_side * 255.0).clip(0, 255).astype(np.uint8)

        if merged_pvetsc is not None:
            metric_config_merged_side = {
                "org": (12, 10 + 2 * text_height + baseline),
                "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale": 0.75,
                "color": (255, 255, 255),
                "thickness": 2,
            }
            cv2.putText(
                merged_side,
                f"PVE-T-SC: {float(merged_pvetsc[0]) * 1000.0:.1f} mm",
                **metric_config_merged_side,
            )

    def _hist_image(data: np.ndarray, title: str, target_hw: tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        fig, ax = plt.subplots(figsize=(w / 100.0, h / 100.0), dpi=100)
        bins = np.linspace(0.0, max_dist, 51)
        counts, edges = np.histogram(data, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        denom = max_dist - 0.0
        if denom <= 0:
            denom = 1.0
        normalized = np.clip((centers - 0.0) / denom, 0.0, 1.0)
        rgba = plt.get_cmap("inferno")(normalized)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=rgba,
            linewidth=0,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Distance (mm)", fontsize=9)
        ax.set_ylabel("Freq", fontsize=9)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.canvas.draw()
        hist_rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        hist_rgb = cv2.cvtColor(hist_rgba, cv2.COLOR_RGBA2RGB)
        return cv2.resize(hist_rgb, (w, h), interpolation=cv2.INTER_AREA)

    # Assemble gallery: top = front views (GT, [merged], per-view),
    # middle = side views, bottom = error histograms (optional).
    top_row_images = [gt_front]
    bottom_row_images = [gt_side]
    hist_row_images = [np.full_like(gt_front, 255)] if plot_hist else None
    if merged_verts is not None:
        top_row_images.append(merged_front_render)
        bottom_row_images.append(merged_side)
        if plot_hist:
            hist_row_images.append(
                _hist_image(
                    merged_vertex_dists,
                    title="Merged hist",
                    target_hw=merged_side.shape[:2],
                )
            )
    for view in range(num_views):
        top_row_images.append(per_view_front[view])
        bottom_row_images.append(per_view_side[view])
        if plot_hist:
            hist_row_images.append(
                _hist_image(
                    per_view_vertex_dists[view],
                    title=f"View {view} hist",
                    target_hw=per_view_side[view].shape[:2],
                )
            )

    top_row = np.concatenate(top_row_images, axis=1)
    bottom_row = np.concatenate(bottom_row_images, axis=1)
    if plot_hist:
        hist_row = np.concatenate(hist_row_images, axis=1)
        gallery_img = np.concatenate([top_row, bottom_row, hist_row], axis=0)
    else:
        gallery_img = np.concatenate([top_row, bottom_row], axis=0)
    gallery_img_bgr = cv2.cvtColor(gallery_img, cv2.COLOR_RGB2BGR)

    # Append error-distance colorbar to the right.
    colorbar_rgb = build_distance_colorbar_rgb(
        min_dist=min_dist,
        max_dist=max_dist,
        cmap="inferno",
        height=gallery_img_bgr.shape[0],
        width=60,
    )
    colorbar_bgr = cv2.cvtColor(colorbar_rgb, cv2.COLOR_RGB2BGR)
    gallery_img_bgr = np.concatenate([gallery_img_bgr, colorbar_bgr], axis=1)

    suffix = "_sc" if sc else ""
    save_path = os.path.join(
        save_dir,
        f"b{batch_idx:03d}_neutral{suffix}.png",
    )
    cv2.imwrite(save_path, gallery_img_bgr)
    logger.info(f"Saved neutral meshes gallery: {save_path}")


# def _count_params(self):
#     # Count and print trainable vs frozen parameters
#     trainable_params = sum(
#         p.numel() for p in self.model.parameters() if p.requires_grad
#     )
#     frozen_params = sum(
#         p.numel() for p in self.model.parameters() if not p.requires_grad
#     )
#     decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
#     if self.use_lora:
#         decoder_lora_params = sum(
#             p.numel() for p in self.model.decoder.lora_layers.parameters()
#         )
#     total_params = trainable_params + frozen_params

#     logger.info("=" * 60)
#     logger.info("Parameter Statistics:")
#     logger.info("=" * 60)
#     logger.info(f"Total parameters: {total_params:,}")
#     logger.info(
#         f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
#     )
#     logger.info(
#         f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)"
#     )
#     logger.info(
#         f"Decoder parameters: {decoder_params:,} ({100 * decoder_params / total_params:.2f}%)"
#     )
#     if self.use_lora:
#         logger.info(
#             f"LoRA decoder parameters: {decoder_lora_params:,} ({100 * decoder_lora_params / total_params:.2f}%)"
#         )
#         # logger.info(f"LoRA trainable parameters: {lora_param_count:,}")
#     logger.info("=" * 60)
