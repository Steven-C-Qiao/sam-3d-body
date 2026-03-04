# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


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

import cv2 
import torch
import matplotlib.cm as cm
def my_visualize(img_cv2, outputs, faces, stack_vertically=True):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

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
    # img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Get mean prediction vertices (original output)
    mean_pred_vertices = person_output["pred_vertices"][0]
    
    # Get samples if available
    vertex_colors = None
    if 'verts_samples' in outputs:
        verts_samples = outputs['verts_samples']
        if isinstance(verts_samples, torch.Tensor):
            verts_samples = verts_samples.cpu().detach().numpy()
        
        # Calculate average distance of each vertex from mean across all samples
        # mhr_samples shape: (batch, num_samples, num_vertices, 3)
        num_samples = verts_samples.shape[1]
        distances = []
        for i in range(num_samples):
            sample_vertices = verts_samples[0, i]  # (num_vertices, 3)
            vertex_distances = np.linalg.norm(sample_vertices - mean_pred_vertices, axis=1)
            distances.append(vertex_distances)
        
        # Average distance across all samples for each vertex
        avg_distances = np.mean(distances, axis=0)  # (num_vertices,)
        
        # Normalize distances to [0, 1] for colormap
        min_dist = np.min(avg_distances)
        max_dist = np.max(avg_distances)
        if max_dist > min_dist:
            normalized_distances = (avg_distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.zeros_like(avg_distances)
        
        # Map to viridis colormap
        viridis = cm.get_cmap('viridis')
        vertex_colors_rgb = viridis(normalized_distances)[:, :3]  # (num_vertices, 3) in [0, 1]
        # Convert to format expected by trimesh: (N, 4) with RGBA in [0, 1] range
        vertex_colors = np.ones((vertex_colors_rgb.shape[0], 4))
        vertex_colors[:, :3] = vertex_colors_rgb

    all_pred_vertices = person_output["pred_vertices"][0] # + person_output["pred_cam_t"][0])
    all_faces = faces
    
    # Pull out a fake translation; take the closest two
    # fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    # all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"][0], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            person_output["pred_cam_t"][0],
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            vertex_colors=vertex_colors,
        )
        * 255
    )

    # Render side view
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
        )
        * 255
    )

    # cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)
    if stack_vertically:
        cur_img = np.concatenate([img_cv2, img_mesh, img_mesh_side], axis=0)
    else:
        cur_img = np.concatenate([img_cv2, img_mesh, img_mesh_side], axis=1)

    return cur_img




def my_visualize_samples(
    img_cv2, 
    outputs, 
    faces, 
    stack_vertically=True, 
    affine=None, 
    img_size=None,
    overlay_gt=True,
    plot_side=True,
    batch=None,
):
    affine = affine.cpu().detach().numpy() if affine is not None else None
    img_size = img_size.cpu().detach().numpy() if img_size is not None else None

    img_mesh = img_cv2.copy()

    # If we have a warp, also precompute the cropped base image
    base_img = img_cv2.copy()
    if affine is not None:
        base_img_uint8 = base_img.astype(np.uint8)
        base_img = cv2.warpAffine(base_img_uint8, affine, img_size)

    mhr_samples = outputs["verts_samples"].cpu().detach().numpy()

    outputs = outputs["mhr"]
    for key in outputs:
        try:
            outputs[key] = outputs[key].cpu().detach().numpy()
        except:
            pass 

    img_mesh_list = []
    img_side_list = []

    all_faces = faces
    renderer = Renderer(
        focal_length=outputs["focal_length"][0], faces=all_faces
    )
    for i in range(mhr_samples.shape[1]):
        img_mesh = img_cv2.copy()
        all_pred_vertices = mhr_samples[0, i]

        # Render front view
        img_mesh = (
            renderer(
                all_pred_vertices,
                outputs["pred_cam_t"][0],
                img_mesh,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        if overlay_gt:
            gt_verts = batch['gt_verts_w_transl'][0].cpu().detach().numpy()
            gt_cam_t = batch["cam_ext"][0][:3, -1].cpu().detach().numpy()
            gt_rgba = renderer(
                gt_verts, gt_cam_t,
                np.ones_like(img_mesh) * 255,
                mesh_base_color=(1.0, 0.8, 0.5),
                scene_bg_color=(1, 1, 1),
                return_rgba=True,
            )
            alpha = (gt_rgba[..., 3:4].astype(np.float32) * 0.5)
            # import ipdb; ipdb.set_trace()
            pred_rgb = img_mesh.astype(np.float32) / 255.0
            gt_rgb = gt_rgba[..., :3].astype(np.float32)
            blended_pred = alpha * gt_rgb + (1.0 - alpha) * pred_rgb
            img_mesh = (blended_pred * 255.0).clip(0, 255).astype(np.uint8)

        if affine is not None:
            img_mesh = cv2.warpAffine(img_mesh, affine, img_size)

        img_mesh_list.append(img_mesh)

        if plot_side:
            pred_side = (
                renderer(
                    all_pred_vertices - all_pred_vertices.mean(axis=0, keepdims=True),
                    np.array([0.0, -0.25, 6.0]),
                    np.ones_like(img_mesh) * 255,
                    mesh_base_color=LIGHT_BLUE,
                    side_view=True,
                    rot_angle=90,
                )
            )
            gt_side = (
                renderer(
                    gt_verts - gt_verts.mean(axis=0, keepdims=True),
                    np.array([0.0, -0.25, 6.0]),
                    np.ones_like(img_mesh) * 255,
                    mesh_base_color=(1.0, 0.8, 0.5),
                    side_view=True,
                    rot_angle=90,
                    return_rgba=True,
                )
            )
            alpha = gt_side[..., 3:4] * 0.5
            gt_side_rgb = gt_side[..., :3]
            blended_sideview = alpha * gt_side_rgb + (1.0 - alpha) * pred_side
            img_side = (blended_sideview * 255.0).clip(0, 255).astype(np.uint8)
            img_side_list.append(img_side)

    axis = 0 if stack_vertically else 1
    img_mesh_list = np.concatenate(img_mesh_list, axis=axis)
    img_side_list = np.concatenate(img_side_list, axis=axis)

    if overlay_gt:
        gt_base_img = (
            renderer(
                gt_verts,
                gt_cam_t,
                img_cv2.copy(),
                mesh_base_color=(1.0, 0.8, 0.5),
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)
        if affine is not None:
            gt_base_img = cv2.warpAffine(gt_base_img, affine, img_size)
    else:
        gt_base_img = base_img

    cur_img = np.concatenate([gt_base_img, img_mesh_list], axis=axis)
    cur_img = np.concatenate([
        cur_img, 
        np.concatenate([gt_base_img, img_side_list], axis=axis), 
    ], axis=1 - axis)

    return cur_img