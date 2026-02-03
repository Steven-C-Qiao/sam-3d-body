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
    if 'mhr_samples' in outputs:
        mhr_samples = outputs['mhr_samples']
        if isinstance(mhr_samples, torch.Tensor):
            mhr_samples = mhr_samples.cpu().detach().numpy()
        
        # Calculate average distance of each vertex from mean across all samples
        # mhr_samples shape: (batch, num_samples, num_vertices, 3)
        num_samples = mhr_samples.shape[1]
        distances = []
        for i in range(num_samples):
            sample_vertices = mhr_samples[0, i]  # (num_vertices, 3)
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

    all_pred_vertices = (person_output["pred_vertices"][0] + person_output["pred_cam_t"][0])
    all_faces = faces
    
    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t

    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"][0], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
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
            fake_pred_cam_t,
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




def my_visualize_samples(img_cv2, outputs, faces, stack_vertically=True):

    img_mesh = img_cv2.copy()

    mhr_samples = outputs['mhr_samples'].cpu().detach().numpy()

    outputs = outputs['mhr']
    for key in outputs:
        try:
            outputs[key] = outputs[key].cpu().detach().numpy()
        except:
            pass
    person_output = outputs 

    img_mesh_list = []
    for i in range(mhr_samples.shape[1]):
        img_mesh = img_cv2.copy()
        all_pred_vertices = (mhr_samples[0, i] + person_output['pred_cam_t'][0])
        all_faces = faces
        
        # Pull out a fake translation; take the closest two
        fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
        all_pred_vertices = all_pred_vertices - fake_pred_cam_t

        # Render front view

        renderer = Renderer(focal_length=person_output["focal_length"][0], faces=all_faces)
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
        img_mesh_list.append(img_mesh)

    if stack_vertically:
        img_mesh_list = np.concatenate(img_mesh_list, axis=0)
        cur_img = np.concatenate([img_cv2, img_mesh_list], axis=0)
    else:
        img_mesh_list = np.concatenate(img_mesh_list, axis=1)
        cur_img = np.concatenate([img_cv2, img_mesh_list], axis=1)

    return cur_img

    
