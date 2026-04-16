"""Geometric measurements on neutral-pose MHR meshes.

Meant for real-image / multi-view pipelines where we want to report
anatomical body measurements from the merged body shape. Inputs are
native MHR neutral-pose coordinates (feet at +y; units: metres).
"""

from typing import Dict, Optional

import numpy as np
import trimesh


def compute_waist_plane_y(
    verts: np.ndarray,
    kp3d: Optional[np.ndarray] = None,
    hip_indices=(9, 10),
    shoulder_indices=(5, 6),
    hip_to_shoulder_frac: float = 0.34,
) -> float:
    """Return the y-coordinate (in ``verts``' frame) of the belly-button /
    natural-waist slicing plane.

    Preferred placement: interpolate between the hip joints and the
    shoulder joints along the body axis, at a fraction
    ``hip_to_shoulder_frac`` from hip toward shoulder. Standard human
    anthropometry places the navel at ~0.60 of body height from the
    ground, i.e. ~0.29 of the hip-to-shoulder distance above the hips.
    We default to 0.34 (roughly 2 cm higher) — this lines up with the
    natural-waist measurement position a tape would sit at on most
    adults. Independent of subject size, so this scales correctly.

    Fallback when ``kp3d`` is absent: 40% from the head toward the feet
    (matches the 0.60-of-height-from-ground heuristic for the navel).
    """
    if kp3d is not None:
        lh, rh = hip_indices
        ls, rs = shoulder_indices
        hip_y = 0.5 * (float(kp3d[lh, 1]) + float(kp3d[rh, 1]))
        shoulder_y = 0.5 * (float(kp3d[ls, 1]) + float(kp3d[rs, 1]))
        # Feet at +y, head at -y, so shoulder_y < hip_y. The linear
        # interpolation ``hip_y + frac * (shoulder_y - hip_y)`` moves
        # toward the head (smaller y) as ``frac`` grows, as intended.
        return hip_y + float(hip_to_shoulder_frac) * (shoulder_y - hip_y)
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    return y_min + 0.40 * (y_max - y_min)


def waist_circumference(
    verts: np.ndarray,
    faces: np.ndarray,
    kp3d: Optional[np.ndarray] = None,
    waist_y: Optional[float] = None,
    hip_indices=(9, 10),
    shoulder_indices=(5, 6),
    hip_to_shoulder_frac: float = 0.34,
    normalize_to_height: Optional[float] = 1.72,
) -> Dict[str, float]:
    """Estimate waist / belly-button circumference by slicing the mesh with a
    horizontal plane.

    Args:
        verts:    (V, 3) neutral-pose MHR vertices. Native MHR convention
                  has feet at +y (up is -y).
        faces:    (F, 3) triangle indices (0-indexed).
        kp3d:     optional (K, 3) MHR keypoints in the same coordinate frame
                  as ``verts``. When provided, the slice plane is placed at
                  ``hip_y + frac * (shoulder_y - hip_y)`` — i.e., a fixed
                  fraction of the hip-to-shoulder distance above the hips
                  (so the landmark scales with subject size).
        waist_y:  explicit y-coordinate for the slice plane. Overrides the
                  keypoint-based placement.
        hip_indices:      (left_hip, right_hip) indices into ``kp3d``
                  (default ``(9, 10)`` — mhr70 convention).
        shoulder_indices: (left_shoulder, right_shoulder) indices into
                  ``kp3d`` (default ``(5, 6)`` — mhr70 convention).
        hip_to_shoulder_frac: belly-button position along the body axis
                  from hip (0) to shoulder (1). Default 0.30 corresponds
                  to standard adult anthropometry.
        normalize_to_height: target total body height in metres. When
                  set (default 1.72 m), the circumference is also returned
                  scaled by ``target / mesh_height`` to factor out
                  subject-size differences. Pass ``None`` to disable.

    Returns dict with keys:
        ``circumference``:      raw waist circumference in metres.
        ``circumference_norm``: height-normalised circumference in metres,
                                or ``None`` when ``normalize_to_height`` is
                                ``None``.
        ``waist_y``:            y-coordinate of the slice plane (same frame
                                as ``verts``). Use this for visualisation.
        ``mesh_height``:        total body height (y_max − y_min) in metres.
    """
    if waist_y is None:
        waist_y = compute_waist_plane_y(
            verts, kp3d=kp3d,
            hip_indices=hip_indices,
            shoulder_indices=shoulder_indices,
            hip_to_shoulder_frac=hip_to_shoulder_frac,
        )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    section = mesh.section(
        plane_origin=np.array([0.0, float(waist_y), 0.0]),
        plane_normal=np.array([0.0, 1.0, 0.0]),
    )

    circumference = 0.0
    torso_bbox_xz = None
    if section is not None:
        # In A/T-pose, the slice often cuts through the arms as well as the
        # torso. Pick only the loop whose (x, z) centroid is closest to the
        # body's vertical axis — that's the torso.
        loops = section.discrete  # list of (N_i, 3) arrays
        if len(loops) > 0:
            best_loop = min(
                loops,
                key=lambda L: float(np.linalg.norm(L[:, [0, 2]].mean(axis=0))),
            )
            edges = np.diff(best_loop, axis=0)
            circumference = float(np.linalg.norm(edges, axis=1).sum())
            # Close the loop if trimesh returned an open polyline.
            if np.linalg.norm(best_loop[0] - best_loop[-1]) > 1e-6:
                circumference += float(np.linalg.norm(best_loop[0] - best_loop[-1]))
            torso_bbox_xz = (
                float(best_loop[:, 0].min()),
                float(best_loop[:, 2].min()),
                float(best_loop[:, 0].max()),
                float(best_loop[:, 2].max()),
            )

    mesh_height = float(verts[:, 1].max() - verts[:, 1].min())

    circum_norm = None
    if normalize_to_height is not None and mesh_height > 0:
        circum_norm = circumference * (float(normalize_to_height) / mesh_height)

    return {
        "circumference": circumference,
        "circumference_norm": circum_norm,
        "waist_y": float(waist_y),
        "mesh_height": mesh_height,
        # (x_min, z_min, x_max, z_max) — axis-aligned bounding box of the
        # torso slice loop in the xz plane. Handy for restricting a visual
        # highlight to the torso (and not the arms) at waist height.
        "torso_bbox_xz": torso_bbox_xz,
    }
