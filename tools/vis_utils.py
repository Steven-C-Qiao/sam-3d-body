# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional

import cv2 
import torch
import numpy as np
import matplotlib.cm as cm

from sam_3d_body.visualization.renderer import Renderer
# from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

# Base pastel mesh color (kept for backward compatibility)
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# High-contrast pair for GT vs predictions (matplotlib/tab10-like)
BLUE = (0.12156863, 0.46666667, 0.70588235)   # strong blue
ORANGE = (1.0,        0.49803922, 0.05490196) # strong orange

# visualizer = SkeletonVisualizer(line_width=2, radius=5)
# visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        * 255
    )

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img



def view_one_in_another(
    outputs, 
    batch,
    mhr_model,
    faces,
    num_views,
    batch_idx=0,
    affine=None,
    img_size=None,
    plot_side=True,
    overlay_gt=True,
):
    """
    Given the prediction body pose, shape and scale from outputs['mhr'], 
    generate a matrix showing how body shape/scale from view i looks when 
    reprojected onto view j using view j's pose predictions.
    
    This is designed for multiview fusion batches where outputs are flattened
    from [bs, num_views, ...] to [bs*num_views, ...].
    
    Args:
        outputs: Model outputs with 'mhr' key containing flattened predictions
        batch: Batch dict with 'img_ori' indexed as [view][batch_idx]
        mhr_model: The MHR model for generating vertices
        faces: Mesh faces
        num_views: Number of views per sample
        batch_idx: Which batch element to visualize (default 0)
        affine: Optional affine transforms [num_views, 2, 3] for cropping
        img_size: Optional target image sizes [num_views, 2] (width, height)
        plot_side: If True, add a second row of 90-degree rotated side-view meshes
        overlay_gt: If True, overlay ground-truth mesh (semi-transparent BLUE) on each cell
    
    Returns:
        cross_view_verts: Tensor of shape [num_views, num_views, num_verts, 3]
            where cross_view_verts[i, j] contains vertices with shape/scale from view i 
            and pose from view j.
        gallery_img: numpy array with the visualization grid
    """
    import torch
    
    mhr_outputs = outputs['mhr']
    
    # Extract parameters for the specified batch element from flattened outputs
    # Flattened index = batch_idx * num_views + view
    start_idx = batch_idx * num_views
    end_idx = start_idx + num_views
    
    body_pose = mhr_outputs['body_pose'][start_idx:end_idx]  # [V, pose_dim]
    shape = mhr_outputs['shape'][start_idx:end_idx]           # [V, 45]
    scale = mhr_outputs['scale'][start_idx:end_idx]           # [V, 68]
    global_rot = mhr_outputs['global_rot'][start_idx:end_idx] # [V, 3]
    hand = mhr_outputs['hand'][start_idx:end_idx]             # [V, hand_dim]
    face = mhr_outputs['face'][start_idx:end_idx]             # [V, face_dim]
    
    # MHR forward config
    mhr_output_config = {
        "return_keypoints": True,
        "return_joint_coords": True,
        "return_model_params": True,
        "return_joint_rotations": True,
        "do_pcblend": True,
    }
    
    # Generate cross-view combinations:
    # For each (i, j) pair, use shape/scale from view i with pose from view j
    cross_view_verts_list = []
    cross_view_root_list = []  # root joint [1, 3] per (i, j) for side-view centering
    
    for i in range(num_views):
        row_verts = []
        row_roots = []
        for j in range(num_views):
            # Shape and scale from view i
            shape_i = shape[i:i+1]  # [1, 45]
            scale_i = scale[i:i+1]  # [1, 68]
            
            # Pose from view j
            body_pose_j = body_pose[j:j+1]   # [1, pose_dim]
            global_rot_j = global_rot[j:j+1] # [1, 3]
            hand_j = hand[j:j+1]             # [1, hand_dim]
            face_j = face[j:j+1]             # [1, face_dim]
            
            # Generate vertices with shape from i and pose from j
            mhr_output = mhr_model.mhr_forward(
                shape_params=shape_i,
                scale_params=scale_i,
                global_trans=torch.zeros_like(global_rot_j),
                global_rot=global_rot_j,
                body_pose_params=body_pose_j,
                hand_pose_params=hand_j,
                expr_params=face_j,
                **mhr_output_config,
            )
            verts_ij, _, jcoords_ij, _, _ = mhr_output
            # Flip Y and Z to match rendering convention
            verts_ij[..., [1, 2]] *= -1
            # Root joint is index 1; flip for consistency
            root_ij = jcoords_ij[:, 1:2, :].clone()  # [1, 1, 3]
            root_ij[..., [1, 2]] *= -1
            
            row_verts.append(verts_ij)  # [1, num_verts, 3]
            row_roots.append(root_ij)   # [1, 1, 3]
        
        # Stack columns for this row: [num_views, num_verts, 3], [num_views, 1, 3]
        row_verts = torch.cat(row_verts, dim=0)
        row_roots = torch.cat(row_roots, dim=0)
        cross_view_verts_list.append(row_verts)
        cross_view_root_list.append(row_roots)
    
    # Stack all rows: [num_views, num_views, num_verts, 3], [num_views, num_views, 1, 3]
    cross_view_verts = torch.stack(cross_view_verts_list, dim=0)
    cross_view_roots = torch.stack(cross_view_root_list, dim=0)
    
    # --- Visualization ---
    # Create a grid where rows = shape source (i), columns = pose source / target image (j)
    # Each cell renders cross_view_verts[i, j] onto view j's image
    
    mhr_outputs = outputs['mhr']
    # Get pred_cam_t for the specified batch element
    pred_cam_t = mhr_outputs['pred_cam_t'][start_idx:end_idx].cpu().detach().numpy()  # [V, 3]
    focal_length = mhr_outputs['focal_length'][start_idx].item()
    
    renderer = Renderer(focal_length=focal_length, faces=faces)
    
    # Get camera intrinsics for proper rendering (flattened)
    cam_int = batch.get('cam_int', None)
    
    # Ground truth for overlay (flattened batch indexing)
    if overlay_gt:
        gt_verts_batch = batch.get("vertices")
        if gt_verts_batch is not None:
            gt_verts_batch = gt_verts_batch.clone()
            gt_verts_batch[..., [1, 2]] *= -1  # un-flip for renderer
        gt_joint_coords_batch = batch.get("joint_coords")
        if "cam_ext" in batch:
            gt_cam_t_batch = batch["cam_ext"][..., :3, -1]
        else:
            gt_cam_t_batch = batch.get("trans_cam")
        has_gt = (
            gt_verts_batch is not None
            and gt_cam_t_batch is not None
            and gt_joint_coords_batch is not None
        )
    else:
        has_gt = False
    
    # Colors for different shape sources (one color per row i)
    shape_colors = [
        # (0.12156863, 0.46666667, 0.70588235),  # blue
        (1.0, 0.49803922, 0.05490196),          # orange
        # (0.17254902, 0.62745098, 0.17254902),  # green
        (0.83921569, 0.15294118, 0.15686275),  # red
        (0.58039216, 0.40392157, 0.74117647),  # purple
        (0.54901961, 0.33725490, 0.29411765),  # brown
        (0.89019608, 0.46666667, 0.76078431),  # pink
        (0.49803922, 0.49803922, 0.49803922),  # gray
    ]
    
    gallery_rows = []
    
    for i in range(num_views):
        row_images = []
        for j in range(num_views):
            # Get target image for view j
            # In multiview batches, img_ori is indexed as [view][batch_idx]
            img_j = batch["img_ori"][j][batch_idx].cpu().detach().numpy()
            
            # Get vertices: shape from i, pose from j
            verts_ij = cross_view_verts[i, j].cpu().detach().numpy()
            cam_t_j = pred_cam_t[j]
            
            # Get camera center if available (flattened index)
            flat_idx_j = batch_idx * num_views + j
            camera_center = None
            if cam_int is not None:
                camera_center = (
                    cam_int[flat_idx_j, 0, 2].item(),
                    cam_int[flat_idx_j, 1, 2].item(),
                )
            
            # Choose color based on shape source
            mesh_color = shape_colors[i % len(shape_colors)]
            
            # Render mesh onto image
            rendered = (
                renderer(
                    verts_ij,
                    cam_t_j,
                    img_j,
                    mesh_base_color=mesh_color,
                    scene_bg_color=(1, 1, 1),
                    camera_center=camera_center,
                )
                * 255
            ).astype(np.uint8)
            
            # Overlay ground truth (semi-transparent BLUE) if requested
            if has_gt:
                gt_verts_j = gt_verts_batch[flat_idx_j]
                gt_cam_t_j = gt_cam_t_batch[flat_idx_j]
                if hasattr(gt_verts_j, "cpu"):
                    gt_verts_j = gt_verts_j.cpu().detach().numpy()
                if hasattr(gt_cam_t_j, "cpu"):
                    gt_cam_t_j = gt_cam_t_j.cpu().detach().numpy()
                gt_rgba = renderer(
                    gt_verts_j,
                    gt_cam_t_j,
                    np.ones_like(img_j) * 255,
                    mesh_base_color=BLUE,
                    scene_bg_color=(1, 1, 1),
                    camera_center=camera_center,
                    return_rgba=True,
                )
                alpha = (gt_rgba[..., 3:4].astype(np.float32) * 0.5)
                pred_rgb = rendered.astype(np.float32) / 255.0
                gt_rgb = gt_rgba[..., :3].astype(np.float32)
                rendered = (alpha * gt_rgb + (1.0 - alpha) * pred_rgb)
                rendered = (rendered * 255.0).clip(0, 255).astype(np.uint8)
            
            # Apply affine transform to crop if provided
            if affine is not None and img_size is not None:
                affine_j = affine[j]
                if hasattr(affine_j, 'cpu'):
                    affine_j = affine_j.cpu().detach().numpy()
                # Ensure affine is 2D (2, 3) and float32
                affine_j = affine_j.reshape(2, 3).astype(np.float32)
                
                img_size_j = img_size[j]
                if hasattr(img_size_j, 'cpu'):
                    img_size_j = img_size_j.cpu().detach().numpy()
                # Flatten and convert to (width, height) tuple
                img_size_j = tuple(img_size_j.flatten().astype(int))
                rendered = cv2.warpAffine(rendered, affine_j, img_size_j)
            
            # Add label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            if i == j:
                label = f"View {j} (own shape)"
            else:
                label = f"Shape {i} -> Pose {j}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            cv2.rectangle(
                rendered,
                (10, 10),
                (10 + text_width + 4, 10 + text_height + baseline + 4),
                bg_color,
                -1,
            )
            cv2.putText(
                rendered,
                label,
                (12, 10 + text_height),
                font,
                font_scale,
                color,
                thickness,
            )
            
            row_images.append(rendered)
        
        # Concatenate columns for this row
        row_img = np.concatenate(row_images, axis=1)
        gallery_rows.append(row_img)
    
    # Optional: 90-degree rotated side-view row (same setup as my_visualize_samples: same renderer/focal_length)
    # Side view tiles use same size as (affine-cropped) front-view tiles so they align; no affine on side.
    if plot_side:
        generic_cam_t = np.array([0.0, -0.25, 6.0])
        # Use same focal length as front view so pred and GT overlay aligns (my_visualize_samples uses one renderer)
        side_renderer = Renderer(focal_length=focal_length, faces=faces)
        
        for i in range(num_views):
            row_images = []
            for j in range(num_views):
                verts_ij = cross_view_verts[i, j].cpu().detach().numpy()
                flat_idx_j = batch_idx * num_views + j
                
                # White background: same size as front-view cell for this column (cropped if affine given)
                if affine is not None and img_size is not None:
                    img_size_j = img_size[j]
                    if hasattr(img_size_j, "cpu"):
                        img_size_j = img_size_j.cpu().detach().numpy()
                    img_size_j = img_size_j.flatten().astype(int)
                    w, h = int(img_size_j[0]), int(img_size_j[1])
                    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                else:
                    img_j = batch["img_ori"][j][batch_idx].cpu().detach().numpy()
                    white_bg = np.ones_like(img_j) * 255
                
                root_ij = cross_view_roots[i, j].cpu().detach().numpy().reshape(3)
                verts_centered = verts_ij - root_ij
                if has_gt:
                    gt_verts_j = gt_verts_batch[flat_idx_j]
                    gt_root_j = gt_joint_coords_batch[flat_idx_j, 1, :]
                    if hasattr(gt_verts_j, "cpu"):
                        gt_verts_j = gt_verts_j.cpu().detach().numpy()
                    if hasattr(gt_root_j, "cpu"):
                        gt_root_j = gt_root_j.cpu().detach().numpy()
                    # gt_verts are in camera space; for 4d-dress they are rotated by cam_ext
                    # but gt_joint_coords are not rotated in preprocess, so put root in same space
                    dataset_name = batch.get("dataset_name", None)
                    if dataset_name is not None:
                        name = dataset_name[0]
                        if isinstance(name, (bytes, np.ndarray)):
                            name = str(name) if not hasattr(name, "item") else name.item()
                        if name == "4d-dress" and "cam_ext" in batch:
                            R = batch["cam_ext"][flat_idx_j, :3, :3]
                            if hasattr(R, "cpu"):
                                R = R.cpu().detach().numpy()
                            gt_root_j = (gt_root_j.reshape(1, 3) @ R.T).reshape(3)
                    gt_centered = gt_verts_j - gt_root_j
                else:
                    gt_centered = None
                
                mesh_color = shape_colors[i % len(shape_colors)]
                rendered = (
                    side_renderer(
                        verts_centered,
                        generic_cam_t,
                        white_bg,
                        mesh_base_color=mesh_color,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                        rot_angle=90,
                    )
                    * 255
                ).astype(np.uint8)
                
                if has_gt and gt_centered is not None:
                    gt_side_rgba = side_renderer(
                        gt_centered,
                        generic_cam_t,
                        white_bg,
                        mesh_base_color=BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                        rot_angle=90,
                        return_rgba=True,
                    )
                    alpha_side = (gt_side_rgba[..., 3:4].astype(np.float32) * 0.5)
                    pred_side_rgb = rendered.astype(np.float32) / 255.0
                    gt_side_rgb = gt_side_rgba[..., :3].astype(np.float32)
                    rendered = (alpha_side * gt_side_rgb + (1.0 - alpha_side) * pred_side_rgb)
                    rendered = (rendered * 255.0).clip(0, 255).astype(np.uint8)
                
                # Label
                if i == j:
                    label = f"View {j} (side)"
                else:
                    label = f"Shape {i} -> Pose {j} (side)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                cv2.rectangle(
                    rendered,
                    (10, 10),
                    (10 + text_width + 4, 10 + text_height + baseline + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    rendered, label, (12, 10 + text_height),
                    font, font_scale, (255, 255, 255), thickness,
                )
                row_images.append(rendered)
            
            row_img = np.concatenate(row_images, axis=1)
            gallery_rows.append(row_img)
    
    # Concatenate all rows (front views, then side views if plot_side)
    gallery_img = np.concatenate(gallery_rows, axis=0)
    
    return cross_view_verts, gallery_img