"""GT-free visualisation helpers for real-image NF-AR pipelines.

Used by scripts/demo_nf_multiview.py. Reuses the pyrender-based Renderer
but does not depend on any GT fields (unlike my_vis.vis_merging_*).

Model reprojection convention (see camera_head.PerspectiveHead.perspective_projection):
``focal_length = cam_int[:, 0, 0]`` and ``pred_cam_t`` / ``pred_vertices``
are expressed in the **full-image** pixel coordinate system, not the
cropped model input. Renders here therefore overlay on the full RGB
image with the full-image ``cam_int`` as camera centre.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ..metrics.mesh_measurements import waist_circumference
from .renderer import Renderer


def _full_image_rgb(batch: Dict, v: int) -> np.ndarray:
    """Full-resolution RGB image for view v (H, W, 3) uint8."""
    return batch["img_ori"][v].data


def _camera_center(cam_int: torch.Tensor, v: int):
    return (float(cam_int[v, 0, 2].item()), float(cam_int[v, 1, 2].item()))


def _grid(tiles: Sequence[np.ndarray], cols: int, pad: int = 4,
          bg: int = 0) -> np.ndarray:
    h, w, _ = tiles[0].shape
    n = len(tiles)
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad,
                      cols * w + (cols + 1) * pad, 3), bg, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = tile
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 14, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _render_overlay(renderer: Renderer, verts: np.ndarray, cam_t: np.ndarray,
                    background: np.ndarray, camera_center,
                    side_view: bool = False) -> np.ndarray:
    """Render a mesh on a background image; return uint8 (H, W, 3)."""
    rendered = renderer(
        verts, cam_t.copy(), background.astype(np.float32),
        camera_center=camera_center, side_view=side_view,
    )
    return np.clip(rendered * 255.0, 0, 255).astype(np.uint8)


def _neutral_render(verts: np.ndarray, faces: np.ndarray,
                    h: int, w: int,
                    cam_t: Tuple[float, float, float] = (0.0, 0.85, 2.5),
                    focal: float = None,
                    highlight_y: Optional[float] = None,
                    highlight_band: float = 0.015,
                    highlight_xz_bbox: Optional[Tuple[float, float, float, float]] = None,
                    highlight_xz_margin: float = 0.03,
                    side_view: bool = False) -> np.ndarray:
    """Render a neutral (rest-pose) mesh against a plain grey background.

    Render resolution is ``(h, w)`` — pass the input image's dims so the
    result can be placed alongside the input in grids without aspect
    stretching. ``focal`` defaults to ``h`` so the mesh occupies a
    consistent fraction of the frame regardless of resolution.

    ``get_mhr_outputs`` leaves neutral verts in MHR (feet-at-+y) coords;
    flip y/z to match the Renderer's convention (see vis_merging_neutral).

    Args:
        highlight_y:        if given, vertices within ``highlight_band``
                            metres of this y value are tinted red — useful
                            for visualising the plane of a measurement.
        highlight_xz_bbox:  optional ``(x_min, z_min, x_max, z_max)`` AABB
                            in the mesh's xz plane. When supplied together
                            with ``highlight_y``, only vertices whose xz
                            lies within this box (plus ``highlight_xz_margin``)
                            are tinted — so the tint follows the torso
                            loop and does NOT spill onto arm cross-sections.
        side_view:          rotate the mesh 90° about y before rendering.
    """
    if focal is None:
        focal = h
    renderer = Renderer(focal_length=focal, faces=faces)
    flipped = verts.copy()
    flipped[..., [1, 2]] *= -1

    vertex_colors = None
    if highlight_y is not None:
        vertex_colors = np.ones((verts.shape[0], 4), dtype=np.float32)
        vertex_colors[:, :3] = np.array([1.0, 1.0, 0.9])  # matches Renderer default
        near_plane = np.abs(verts[:, 1] - highlight_y) < highlight_band
        if highlight_xz_bbox is not None:
            x_min, z_min, x_max, z_max = highlight_xz_bbox
            m = float(highlight_xz_margin)
            in_bbox = (
                (verts[:, 0] >= x_min - m) & (verts[:, 0] <= x_max + m)
                & (verts[:, 2] >= z_min - m) & (verts[:, 2] <= z_max + m)
            )
            near_plane = near_plane & in_bbox
        vertex_colors[near_plane, :3] = np.array([1.0, 0.25, 0.25])  # red band

    bg = np.full((h, w, 3), 180, dtype=np.uint8)
    rendered = renderer(flipped, np.array(cam_t, dtype=np.float32),
                        bg.astype(np.float32), vertex_colors=vertex_colors,
                        side_view=side_view, rot_angle=90)
    return np.clip(rendered * 255.0, 0, 255).astype(np.uint8)


def _fit_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _tile_size(img_h: int, img_w: int, display_height: int) -> Tuple[int, int]:
    """Target tile size preserving the input aspect ratio."""
    scale = display_height / img_h
    return display_height, max(1, int(round(img_w * scale)))


def _stack_pair(top: np.ndarray, bottom: np.ndarray, pad: int = 4) -> np.ndarray:
    sep = np.zeros((pad, top.shape[1], 3), dtype=np.uint8)
    return np.concatenate([top, sep, bottom], axis=0)


def vis_per_view_samples(
    outputs: Dict,
    batch: Dict,
    faces: np.ndarray,
    save_dir: str,
    max_samples: int = 8,
    cols: int = 4,
    display_height: int = 540,
) -> None:
    """Per view, save a grid of front/side pairs.

    Each column shows a (front, side) mesh render pair for the input /
    mean / sample{k}. The front-view overlay is on the input image; the
    side-view (90° about Y) is on a plain grey background. Columns wrap
    every ``cols`` pairs.
    """

    verts_samples = outputs["verts_samples"].detach().cpu().numpy()
    mhr = outputs["mhr"]
    mean_verts = mhr["pred_vertices"].detach().cpu().numpy()
    mean_cam_t = mhr["pred_cam_t"].detach().cpu().numpy()
    focals = mhr["focal_length"].detach().cpu().numpy().reshape(-1)
    cam_int = batch["cam_int"]
    cam_t_samples = outputs.get("pred_cam_t_samples")
    if cam_t_samples is not None:
        cam_t_samples = cam_t_samples.detach().cpu().numpy()

    num_views = verts_samples.shape[0]
    S = min(verts_samples.shape[1], max_samples)

    for v in range(num_views):
        bg = _full_image_rgb(batch, v)
        h, w, _ = bg.shape
        center = _camera_center(cam_int, v)
        renderer = Renderer(focal_length=float(focals[v]), faces=faces)
        side_bg = np.full_like(bg, 180)

        def _front(verts, ct):
            return _render_overlay(renderer, verts, ct, bg, center, side_view=False)

        def _side(verts, ct):
            return _render_overlay(renderer, verts, ct, side_bg, center, side_view=True)

        mean_front = _front(mean_verts[v], mean_cam_t[v])
        mean_side = _side(mean_verts[v], mean_cam_t[v])
        sample_fronts, sample_sides = [], []
        for k in range(S):
            ct = cam_t_samples[v, k] if cam_t_samples is not None else mean_cam_t[v]
            sample_fronts.append(_front(verts_samples[v, k], ct))
            sample_sides.append(_side(verts_samples[v, k], ct))

        tile_h, tile_w = _tile_size(h, w, display_height)
        blank = np.full((tile_h, tile_w, 3), 30, dtype=np.uint8)

        # Build (front, side, label) triples.
        entries = [
            (_fit_to(bg, tile_h, tile_w), blank, "input"),
            (_fit_to(mean_front, tile_h, tile_w),
             _fit_to(mean_side, tile_h, tile_w), "mean"),
        ]
        for k in range(S):
            entries.append((
                _fit_to(sample_fronts[k], tile_h, tile_w),
                _fit_to(sample_sides[k], tile_h, tile_w),
                f"sample{k}",
            ))

        # Each entry becomes a single tile: front on top, side below.
        pairs: List[np.ndarray] = []
        for front, side, name in entries:
            top_lbl = f"{name} (front)" if name != "input" else "input"
            bot_lbl = f"{name} (side)" if name != "input" else ""
            top = _label(front, top_lbl)
            bot = _label(side, bot_lbl) if bot_lbl else side
            pairs.append(_stack_pair(top, bot))

        grid = _grid(pairs, cols=cols)
        out_path = f"{save_dir}/per_view_samples_{v:02d}.png"
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  wrote {out_path}")


def vis_merged(
    outs: Dict,
    outputs: Dict,
    batch: Dict,
    faces: np.ndarray,
    save_dir: str,
    merged_cam_t: torch.Tensor = None,
    display_height: int = 540,
) -> None:
    """Render the merged mesh (shared shape, per-view pose) onto each view
    and a neutral-pose comparison. Neutral renders share the aspect ratio
    of the input image to avoid stretching in the summary grid.

    ``merged_cam_t`` (B*V, 3): per-view camera translation resampled from
    stage 2 conditioned on the merged shape (see
    ``nf_merging.resample_cam_for_merged_shape``). When provided, used for
    rendering the merged mesh so the reprojection matches the merged body
    size; falls back to ``outputs["mhr"]["pred_cam_t"]`` otherwise.
    """

    mhr = outputs["mhr"]
    mean_cam_t = mhr["pred_cam_t"].detach().cpu().numpy()
    if merged_cam_t is None:
        merged_cam_t_np = mean_cam_t
    else:
        merged_cam_t_np = merged_cam_t.detach().cpu().numpy()
    focals = mhr["focal_length"].detach().cpu().numpy().reshape(-1)
    cam_int = batch["cam_int"]
    merged_verts = outs["merged_verts"].detach().cpu().numpy()
    avg_verts = outs["avg_verts"].detach().cpu().numpy()
    merged_neutral = outs["merged_neutral_verts"].detach().cpu().numpy()
    per_view_neutral = outs["per_view_neutral_verts"].detach().cpu().numpy()
    merged_neutral_kp3d = outs["merged_neutral_kp3d"].detach().cpu().numpy()
    per_view_neutral_kp3d = outs["per_view_neutral_kp3d"].detach().cpu().numpy()
    best_sample_neutral = outs.get("best_logprob_sample_neutral_verts")
    best_sample_kp3d = outs.get("best_logprob_sample_neutral_kp3d")
    if best_sample_neutral is not None:
        best_sample_neutral = best_sample_neutral.detach().cpu().numpy()
    if best_sample_kp3d is not None:
        best_sample_kp3d = best_sample_kp3d.detach().cpu().numpy()

    num_views = merged_verts.shape[0]

    target_height_m = 1.72

    def _waist_measure(verts: np.ndarray, kp3d: np.ndarray):
        return waist_circumference(
            verts, faces, kp3d=kp3d, normalize_to_height=target_height_m,
        )

    def _waist_label(name: str, meas: Dict[str, float]) -> str:
        # Normalised so the subject's body height = target_height_m.
        return (
            f"{name} | waist {meas['circumference_norm'] * 100:.1f} cm "
            f"@ {int(target_height_m * 100)}cm"
        )

    summary_rows = []
    for v in range(num_views):
        bg = _full_image_rgb(batch, v)
        h, w, _ = bg.shape
        center = _camera_center(cam_int, v)
        posed_renderer = Renderer(focal_length=float(focals[v]), faces=faces)
        tile_h, tile_w = _tile_size(h, w, display_height)

        mean_posed = _render_overlay(
            posed_renderer, avg_verts[v], mean_cam_t[v], bg, center
        )
        merged_posed = _render_overlay(
            posed_renderer, merged_verts[v], merged_cam_t_np[v], bg, center
        )

        # Measure each neutral mesh; compute the waist plane once per mesh
        # so we can both label the tile AND highlight the slice in the render.
        merged_meas = _waist_measure(merged_neutral[v], merged_neutral_kp3d[v])
        per_view_meas = _waist_measure(
            per_view_neutral[v], per_view_neutral_kp3d[v]
        )
        best_meas = None
        if best_sample_neutral is not None:
            best_meas = _waist_measure(
                best_sample_neutral[v], best_sample_kp3d[v]
            )

        def _neu(verts_, meas_, side=False):
            return _neutral_render(
                verts_, faces, tile_h, tile_w,
                highlight_y=meas_["waist_y"],
                highlight_xz_bbox=meas_["torso_bbox_xz"],
                side_view=side,
            )

        merged_neu_img = _neu(merged_neutral[v], merged_meas)
        merged_neu_side = _neu(merged_neutral[v], merged_meas, side=True)
        per_view_neu_img = _neu(per_view_neutral[v], per_view_meas)
        per_view_neu_side = _neu(per_view_neutral[v], per_view_meas, side=True)
        best_neu_img = best_neu_side = None
        if best_sample_neutral is not None:
            best_neu_img = _neu(best_sample_neutral[v], best_meas)
            best_neu_side = _neu(best_sample_neutral[v], best_meas, side=True)

        posed_tiles = [
            _label(_fit_to(bg, tile_h, tile_w), "input"),
            _label(_fit_to(mean_posed, tile_h, tile_w), "mean (posed)"),
            _label(_fit_to(merged_posed, tile_h, tile_w), "merged (posed)"),
        ]
        posed_grid = _grid(posed_tiles, cols=3)
        cv2.imwrite(f"{save_dir}/merged_posed_{v:02d}.png",
                    cv2.cvtColor(posed_grid, cv2.COLOR_RGB2BGR))

        per_view_lbl = _waist_label("per-view mean (neutral)", per_view_meas)
        merged_lbl = _waist_label("merged (neutral)", merged_meas)
        neutral_tiles = [
            _label(per_view_neu_img, per_view_lbl),
            _label(per_view_neu_side, "per-view (side)"),
            _label(merged_neu_img, merged_lbl),
            _label(merged_neu_side, "merged (side)"),
        ]
        if best_neu_img is not None:
            best_lbl = _waist_label("best-logprob sample (neutral)", best_meas)
            neutral_tiles.append(_label(best_neu_img, best_lbl))
            neutral_tiles.append(_label(best_neu_side, "best sample (side)"))
        neutral_grid = _grid(neutral_tiles, cols=len(neutral_tiles))
        cv2.imwrite(f"{save_dir}/merged_neutral_{v:02d}.png",
                    cv2.cvtColor(neutral_grid, cv2.COLOR_RGB2BGR))

        row_tiles = [
            _label(_fit_to(bg, tile_h, tile_w), f"v{v:02d} input"),
            _label(_fit_to(mean_posed, tile_h, tile_w), "mean (posed)"),
            _label(_fit_to(merged_posed, tile_h, tile_w), "merged (posed)"),
            _label(merged_neu_img, merged_lbl),
            _label(merged_neu_side, "merged (side)"),
            _label(per_view_neu_img, per_view_lbl),
            _label(per_view_neu_side, "per-view (side)"),
        ]
        summary_rows.append(_grid(row_tiles, cols=len(row_tiles)))

    max_w = max(r.shape[1] for r in summary_rows)
    pad = 4
    padded = []
    for r in summary_rows:
        if r.shape[1] < max_w:
            pad_right = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad_right], axis=1)
        padded.append(r)
        padded.append(np.zeros((pad, max_w, 3), dtype=np.uint8))
    summary = np.concatenate(padded[:-1], axis=0)
    cv2.imwrite(f"{save_dir}/merged_summary.png",
                cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    print(f"  wrote merged_summary.png")
