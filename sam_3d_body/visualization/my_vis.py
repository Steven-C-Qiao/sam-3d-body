import torch
import os

import matplotlib

matplotlib.use("Agg")  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np
import pytorch_lightning as pl
import matplotlib.colors

# import scenepic as sp
from einops import rearrange
from collections import defaultdict

# Pyrender imports for mesh rendering
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


try:
    import pyvista as pv

    PV_AVAILABLE = True
    # Set PyVista to use offscreen rendering
    pv.OFF_SCREEN = True
except ImportError:
    PV_AVAILABLE = False


class Visualiser(pl.LightningModule):
    def __init__(self, save_dir, cfg=None, rank=0, faces=None):
        super().__init__()
        self.save_dir = save_dir
        self.rank = rank
        self.cfg = cfg
        self._suffix = ""
        self.faces = faces  # Store faces for mesh rendering

        # self.threshold = 50
        # print("Visualiser confidence threshold:", self.threshold)

    def set_global_rank(self, global_rank):
        self.rank = global_rank

    def _get_filename(self, suffix=""):
        """
        Generate filename with format: {counter:06d}_{epoch:03d}_{split}{suffix}.png
        """
        split_part = f"_{self._split}" if self._split else ""
        return f"{self.counter:06d}_{self._epoch:03d}{split_part}{suffix}.png"

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

        self.visualise_full(predictions, batch)

    def visualise_full(self, predictions, batch):
        batch["keypoints_3d"][..., [1, 2]] *= -1
        predictions["mhr"]["pred_keypoints_3d"][..., [1, 2]] *= -1
        predictions["mhr_samples_keypoints_3d"][..., [1, 2]] *= -1
        predictions["mhr_samples"][..., [1, 2]] *= -1
        predictions["mhr"]["pred_vertices"][..., [1, 2]] *= -1

        self.visualise_keypoints_3d(predictions, batch)

        self.visualise_2d_keypoints_full(predictions, batch)

        self.visualise_2d_keypoints_cropped(predictions, batch)

        self.visualise_mesh(predictions, batch)

        self.visualise_mesh_pyplot(predictions, batch)

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
        pred_keypoints_3d_samples = predictions.get("mhr_samples_keypoints_3d", None)

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
        pred_verts_samples = predictions["mhr_samples"][0]  # (num_samples, 18439, 3)

        # Convert to numpy if tensor
        if isinstance(gt_verts, torch.Tensor):
            gt_verts = gt_verts.cpu().detach().numpy()
        if isinstance(pred_verts, torch.Tensor):
            pred_verts = pred_verts.cpu().detach().numpy()
        if isinstance(pred_verts_samples, torch.Tensor):
            pred_verts_samples = pred_verts_samples.cpu().detach().numpy()

        # Create figure with subplots
        num_samples = pred_verts_samples.shape[0]
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
        pred_verts_samples = predictions["mhr_samples"][0]  # (num_samples, 18439, 3)

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
        if "mhr_samples_keypoints_2d" not in predictions:
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

        # Ensure img_size is a 2-element array
        if isinstance(img_size, np.ndarray):
            if img_size.shape == ():
                img_size = np.array([img_size, img_size])
            elif len(img_size.shape) == 1 and len(img_size) >= 2:
                img_size = img_size[:2]
            else:
                img_size = np.array([256, 256])
        else:
            img_size = np.array([256, 256])

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
        sample_kp2d_full = predictions["mhr_samples_keypoints_2d"][
            batch_idx
        ]  # [num_samples, 70, 2]
        num_samples = sample_kp2d_full.shape[0]

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
        """
        Visualize ground truth, predicted mean, and sampled 2D keypoints on the cropped image.

        Args:
            predictions: Model predictions dictionary (already converted to numpy)
            batch: Input batch dictionary (already converted to numpy)
        """
        if "mhr_samples_keypoints_2d" not in predictions:
            logger.warning(
                "No sample keypoints found in predictions. Skipping 2D keypoint visualization."
            )
            return

        # For consistency with other visualisation functions, only visualise the first element in the batch
        batch_idx = 0

        # Extract keypoints for this batch item
        # GT keypoints are now normalized to [-0.5, 0.5] in cropped coordinate space
        gt_kp2d_normalized = batch["keypoints_2d"][
            batch_idx, :, :
        ]  # [N, 2] in normalized coords [-0.5, 0.5]
        # Unnormalize GT to pixel coordinates [0, 256]
        gt_kp2d = (gt_kp2d_normalized + 0.5) * 256  # [N, 2]

        # Predicted keypoints in cropped normalized coords [-0.5, 0.5]
        pred_kp2d_cropped_normalised = predictions["mhr"]["pred_keypoints_2d_cropped"][
            batch_idx
        ]  # [N, 2]
        # Convert to pixel coordinates
        pred_kp2d_cropped_coords = (pred_kp2d_cropped_normalised + 0.5) * 256  # [N, 2]

        # Use pre-computed normalized cropped sample keypoints if available
        if "mhr_samples_keypoints_2d_cropped" in predictions:
            sample_kp2d_cropped_normalized = predictions[
                "mhr_samples_keypoints_2d_cropped"
            ][
                batch_idx
            ]  # [num_samples, N, 2]
            num_samples = sample_kp2d_cropped_normalized.shape[0]
            # Unnormalize to pixel coordinates [0, 256]
            sample_kp2d_cropped_coords = (
                sample_kp2d_cropped_normalized + 0.5
            ) * 256  # [num_samples, N, 2]
        else:
            # Fallback: convert from full image coordinates (old method)
            sample_kp2d_full = predictions["mhr_samples_keypoints_2d"][
                batch_idx
            ]  # [num_samples, N, 2]
            num_samples = sample_kp2d_full.shape[0]
            sample_kp2d_cropped_coords = []

            # Get affine transformation and image size from batch
            if "affine_trans" in batch and "img_size" in batch:
                try:
                    affine_trans = batch["affine_trans"][
                        batch_idx, 0
                    ]  # [2, 3] or [3, 3]
                    img_size = batch["img_size"][batch_idx, 0]  # [2] (width, height)

                    # Ensure img_size is a 2-element array
                    if isinstance(img_size, np.ndarray):
                        if img_size.shape == ():
                            img_size = np.array([img_size, img_size])
                        elif len(img_size.shape) == 1 and len(img_size) >= 2:
                            img_size = img_size[:2]
                        else:
                            img_size = np.array([256, 256])  # Default
                    else:
                        img_size = np.array([256, 256])  # Default

                    # Convert sample keypoints from full image coords to cropped coords
                    for i in range(num_samples):
                        N_kp = sample_kp2d_full.shape[1]
                        sample_kp2d_homogeneous = np.ones((N_kp, 3))
                        sample_kp2d_homogeneous[:, :2] = sample_kp2d_full[i]  # [N, 2]

                        # Apply affine transformation
                        if affine_trans.shape == (2, 3):
                            sample_kp2d_cropped = (
                                sample_kp2d_homogeneous @ affine_trans.T
                            )  # [N, 2]
                        elif affine_trans.shape == (3, 3):
                            sample_kp2d_transformed = (
                                sample_kp2d_homogeneous @ affine_trans.T
                            )  # [N, 3]
                            sample_kp2d_cropped = sample_kp2d_transformed[
                                :, :2
                            ]  # [N, 2]
                        else:
                            sample_kp2d_cropped = sample_kp2d_full[i]

                        # Normalize to [-0.5, 0.5] then unnormalize to pixel coords
                        sample_kp2d_cropped_normalized = (
                            sample_kp2d_cropped / img_size.reshape(1, 2) - 0.5
                        )
                        sample_kp2d_cropped_pixel = (
                            sample_kp2d_cropped_normalized + 0.5
                        ) * 256
                        sample_kp2d_cropped_coords.append(sample_kp2d_cropped_pixel)

                    sample_kp2d_cropped_coords = np.array(
                        sample_kp2d_cropped_coords
                    )  # [num_samples, N, 2]
                except Exception as e:
                    logger.warning(
                        f"Failed to convert sample keypoints to cropped coords: {e}. Skipping samples."
                    )
                    sample_kp2d_cropped_coords = None
            else:
                logger.warning(
                    "affine_trans or img_size not found in batch. Skipping sample keypoint conversion."
                )
                sample_kp2d_cropped_coords = None

        # Get cropped image
        img = batch["img"][batch_idx, 0]  # [3, 256, 256] or [256, 256, 3]
        if img.shape[0] == 3:
            # CHW format, convert to HWC
            img = img.transpose(1, 2, 0)
        # Normalize if needed (assuming image is in [0, 1] or normalized range)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Create visualization
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
