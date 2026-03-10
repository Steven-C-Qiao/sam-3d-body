import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


DATA_BASE_PATH = "/scratch/cq244/BEDLAM/data/"
NPZ_PATH = os.path.join(
    DATA_BASE_PATH,
    "training_labels/all_npz_12_training_extra_mhr/20221010_3_1000_batch01hand_6fps.npz",
)
IMAGE_DIR = os.path.join(
    DATA_BASE_PATH, "training_images/20221010_3_1000_batch01hand_6fps"
)
MHR_MODEL_PATH = (
    "/scratch/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)

        # If we have more than 3 channels, return all channels
        if texels.shape[-1] > 3:
            # Use the alpha channel from fragments for masking
            alpha = fragments.zbuf[..., 0] > -1

            # Expand alpha to match number of channels
            alpha = alpha.unsqueeze(-1).unsqueeze(
                -1
            )  # no need to expand as texels as same alpha for all channels

            # Return all channels with alpha mask
            return torch.cat([texels, alpha], dim=-1).squeeze()  # (N, H, W, C+1)

        # For RGB textures, use standard blending
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image for RGB, or (N, H, W, C+1) for multi-channel


class FeatureRenderer(pl.LightningModule):
    def __init__(self, image_size=(256, 192)):
        super().__init__()
        self.image_size = image_size

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=2,
            bin_size=None,
            max_faces_per_bin=40000, 
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SimpleShader(),
        )
        # self.register_buffer('faces', smpl_faces)

        self._set_cameras(PerspectiveCameras().to(self.device))

    def forward(self, mesh, render_img=False, **kwargs):
        if render_img:
            images = self.renderer(mesh)
            maps = images[..., :-1]
            mask = images[..., -1]
        else:
            maps = None
            mask = None

        fragments = self.renderer.rasterizer(mesh)

        ret = {
            'maps': maps,
            'mask': mask,
            "fragments": fragments,
        }

        return ret

    def _set_cameras(self, cameras):
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras


def project(points, cam_trans, cam_int, return_depth=True):
    points = points + cam_trans
    depth = points[..., -1]
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.einsum("bij, bkj->bki", cam_int, projected_points)
    if return_depth:
        return projected_points, depth
    else:
        return projected_points


def check_vertex_visibility(
    v2d,
    v2d_depth,
    fragments,
    depth_tolerance=0.01,
):
    """
    Simple vertex visibility check using z-buffer.
    A vertex is visible if:
    1. It projects within image bounds
    2. The z-buffer depth at its projected location matches the vertex depth (within tolerance)
    
    Args:
        v2d: (N, V, 2) - projected vertex 2D coordinates
        v2d_depth: (N, V) - vertex depths
        fragments: PyTorch3D fragments object
        depth_tolerance: float - tolerance for depth matching
    Returns:
        visibility: (N, V) - boolean visibility mask
    """
    zbuf = fragments.zbuf  # (N, H, W, K)
    zbuf = torch.flip(zbuf, dims=[1, 2])  # NOTE: Flip here similar to the image
    N, H, W, K = zbuf.shape

    # v2d: (N, V, 2)
    x_pixel, y_pixel = v2d[..., 0], v2d[..., 1]  # (N, V)

    # Round to integer pixel coordinates for indexing
    x_idx = torch.clamp(torch.round(x_pixel).long(), 0, W - 1)
    y_idx = torch.clamp(torch.round(y_pixel).long(), 0, H - 1)

    in_bounds = (x_pixel >= 0) & (x_pixel < W) & (y_pixel >= 0) & (y_pixel < H)

    # Get the closest depth from z-buffer (first face, index 0)
    batch_idx = torch.arange(N, device=zbuf.device)[:, None].expand(-1, y_idx.shape[1])  # (N, V)
    zbuf_depth_at_pixels = zbuf[batch_idx, y_idx, x_idx, 0]  # (N, V) - use first face

    # Check if z-buffer has valid depth (not -1)
    valid_zbuf = zbuf_depth_at_pixels > -1

    # Check if vertex depth matches z-buffer depth (within tolerance)
    # Vertex is visible if its depth is close to the rendered depth
    depth_match = torch.abs(zbuf_depth_at_pixels - v2d_depth) < depth_tolerance

    visibility = in_bounds & valid_zbuf & depth_match
    return visibility


def process_npz_paths_on_gpu(npz_paths, gpu_id, mhr_model_path, ckpt_path, debug_visualize=False):
    """
    Process a list of npz_paths on a specific GPU.
    
    Args:
        npz_paths: list of npz file paths to process
        gpu_id: int - GPU ID to use (0, 1, 2, or 3)
        mhr_model_path: str - path to MHR model
        ckpt_path: str - path to checkpoint file
        debug_visualize: bool - whether to create debug visualizations
    """
    # Set the GPU for this process
    # Note: In a new process, we can set CUDA_VISIBLE_DEVICES and it will work
    # since PyTorch hasn't initialized CUDA in this process yet
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use the first (and only) visible GPU
    
    print(f"[GPU {gpu_id}] Starting processing {len(npz_paths)} files on device {device}")
    
    # Load model and checkpoint on this GPU
    mhr_model = torch.jit.load(
        mhr_model_path,
        map_location=device,
    )
    
    ckpt = torch.load(ckpt_path, weights_only=False)
    faces = ckpt["head_pose.faces"].cpu().detach().numpy()
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for npz_idx, npz_path in enumerate(npz_paths):
            path_if_already_done = npz_path[:-4] + "_vertex_visibility.npz"
            if os.path.exists(path_if_already_done):
                print(f'[GPU {gpu_id}] {npz_path} already done')
                continue

            print(f'[GPU {gpu_id}] Processing {npz_path} ({npz_idx+1}/{len(npz_paths)})')
            
            vertices = []
            data = np.load(npz_path)

            new_data = {}
            for key in data.keys():
                try:
                    new_data[key] = torch.tensor(data[key], dtype=torch.float32, device=device)
                except:
                    print(f'[GPU {gpu_id}] {key} not converted to tensor')

            data = new_data

            chunk_size = 64
            num_samples = data["identity_coeffs"].shape[0]
            for i in tqdm(range(0, num_samples, chunk_size), desc=f"[GPU {gpu_id}] Generating vertices"):
                end_idx = min(i + chunk_size, num_samples)

                verts, skeleton = mhr_model(
                    data["identity_coeffs"][i:end_idx],
                    data["lbs_model_params"][i:end_idx],
                    data["face_expr_coeffs"][i:end_idx],
                )
                verts /= 100.0
                vertices.append(verts)

            vertices = torch.cat(vertices, dim=0)

            closeup = "closeup" in npz_path.lower()
            image_size = (720, 1280) if not closeup else (1280, 720)

            downscale_by = 2
            image_size = (int(image_size[0] / downscale_by), int(image_size[1] / downscale_by))
            
            renderer = FeatureRenderer(image_size=image_size)
            renderer = renderer.to(device)

            all_vertex_visibilities = []
            for i in tqdm(range(0, num_samples, chunk_size), desc=f"[GPU {gpu_id}] Computing vertex visibility"):
                
                end_idx = min(i + chunk_size, num_samples)

                verts = vertices[i:end_idx]

                trans_cam = data["trans_cam"][i:end_idx]
                cam_t = data["cam_ext"][i:end_idx, :3, -1]
                cam_t += trans_cam

                cam_int = data["cam_int"][i:end_idx]
                cam_int[:, 0, 0] /= downscale_by
                cam_int[:, 1, 1] /= downscale_by
                cam_int[:, 0, 2] /= downscale_by
                cam_int[:, 1, 2] /= downscale_by
                focal_length = cam_int[:, 0, 0]
                cam_center = cam_int[:, [0, 1], 2]

                # Project all vertices to 2D
                v2d, v2d_depth = project(
                    verts,
                    cam_t[:, None],
                    cam_int,
                    return_depth=True,
                )

                cameras = PerspectiveCameras(
                    focal_length=focal_length[:, None],
                    principal_point=cam_center,
                    T=cam_t,
                    image_size=[image_size],
                    in_ndc=False,
                    device=device,
                )
                renderer._set_cameras(cameras)

                textures = torch.ones_like(
                    verts, dtype=torch.float32, device=device
                ) * torch.tensor([0.6, 0.8, 1.0], dtype=torch.float32, device=device)

                mesh = Meshes(
                    verts=verts,
                    faces=faces.unsqueeze(0).repeat(verts.shape[0], 1, 1),
                    textures=TexturesVertex(verts_features=textures),
                )

                pytorch3d_output = renderer(mesh)

                fragments = pytorch3d_output["fragments"]

                # Compute visibility for all vertices
                vertex_visibility = check_vertex_visibility(
                    v2d,
                    v2d_depth,
                    fragments=fragments,
                    depth_tolerance=0.01,  # Adjust based on your scale
                ).cpu().numpy()

                all_vertex_visibilities.append(vertex_visibility)
                
                # Debug visualization: visualize first sample of first chunk
                # if debug_visualize and i == 0:
                #     vis_save_path = os.path.join(
                #         os.path.dirname(npz_path),
                #         f"{os.path.basename(npz_path)[:-4]}_vertex_visibility_3d_sample0.png"
                #     )
                #     visualize_vertex_visibility_3d(
                #         verts,
                #         vertex_visibility,
                #         sample_idx=0,
                #         save_path=vis_save_path
                #     )

            all_vertex_visibilities = np.concatenate(all_vertex_visibilities, axis=0)
            np.savez(
                os.path.join(npz_path[:-4] + "_vertex_visibility.npz"), 
                vertex_visibility=all_vertex_visibilities
            )
            print(f'[GPU {gpu_id}] Saved vertex visibility labels to {npz_path[:-4] + "_vertex_visibility.npz"}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"[GPU {gpu_id}] Finished processing all {len(npz_paths)} files")


def visualize_vertex_visibility_3d(vertices, visibility, sample_idx=0, save_path=None):
    """
    Create a 3D scatter plot showing vertex visibilities.
    
    Args:
        vertices: (N, V, 3) - vertex positions
        visibility: (N, V) - boolean visibility mask
        sample_idx: int - which sample to visualize
        save_path: str - path to save the figure (if None, displays interactively)
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(visibility, torch.Tensor):
        visibility = visibility.cpu().numpy()
    
    # Get vertices and visibility for the specified sample
    verts = vertices[sample_idx]  # (V, 3)
    vis = visibility[sample_idx]  # (V,)
    
    # Create figure with 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate visible and invisible vertices
    visible_verts = verts[vis]
    invisible_verts = verts[~vis]
    
    # Plot visible vertices in green
    if len(visible_verts) > 0:
        ax.scatter(
            visible_verts[:, 0],
            visible_verts[:, 1],
            visible_verts[:, 2],
            c='green',
            s=1,
            alpha=0.6,
            label=f'Visible ({len(visible_verts)})'
        )
    
    # Plot invisible vertices in red
    if len(invisible_verts) > 0:
        ax.scatter(
            invisible_verts[:, 0],
            invisible_verts[:, 1],
            invisible_verts[:, 2],
            c='red',
            s=1,
            alpha=0.6,
            label=f'Invisible ({len(invisible_verts)})'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Vertex Visibility Visualization (Sample {sample_idx})\n'
                 f'Total: {len(verts)}, Visible: {len(visible_verts)}, Invisible: {len(invisible_verts)}')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min()
    ]).max() / 2.0
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for proper CUDA isolation
    # 'spawn' creates a fresh Python interpreter in each process, ensuring
    # CUDA_VISIBLE_DEVICES works correctly
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # If start method already set, continue (shouldn't happen with force=True)
        pass
    
    # Debug flag: set to True to enable 3D visualization of vertex visibilities
    DEBUG_VISUALIZE = False
    
    # Number of GPUs to use
    NUM_GPUS = 4
    
    CKPT_PATH = "/scratch/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    
    # Get all npz files to process
    all_npzs = sorted([
        os.path.join(DATA_BASE_PATH, "training_labels/all_npz_12_training_extra_mhr", f)
        for f in os.listdir(
            os.path.join(
                DATA_BASE_PATH, "training_labels/all_npz_12_training_extra_mhr"
            )
        )
        if (f.endswith(".npz") and not f.endswith("_visibility.npz") and not f.endswith("_visibility_308.npz") and not f.endswith("_vertex_visibility.npz"))
    ])
    
    print(f"Found {len(all_npzs)} files to process")
    
    if len(all_npzs) == 0:
        print("No files to process. Exiting.")
        exit(0)
    
    # Split npz_paths into chunks for each GPU
    chunk_size = (len(all_npzs) + NUM_GPUS - 1) // NUM_GPUS  # Ceiling division
    npz_chunks = [all_npzs[i:i + chunk_size] for i in range(0, len(all_npzs), chunk_size)]
    
    # Ensure we have exactly NUM_GPUS chunks (pad with empty lists if needed)
    while len(npz_chunks) < NUM_GPUS:
        npz_chunks.append([])
    
    print(f"Split into {len(npz_chunks)} chunks:")
    for i, chunk in enumerate(npz_chunks):
        print(f"  GPU {i}: {len(chunk)} files")
    
    # Create processes for each GPU
    processes = []
    for gpu_id in range(NUM_GPUS):
        if len(npz_chunks[gpu_id]) > 0:
            p = mp.Process(
                target=process_npz_paths_on_gpu,
                args=(
                    npz_chunks[gpu_id],
                    gpu_id,
                    MHR_MODEL_PATH,
                    CKPT_PATH,
                    DEBUG_VISUALIZE,
                )
            )
            p.start()
            processes.append(p)
            print(f"Started process for GPU {gpu_id} (PID: {p.pid})")
        else:
            print(f"Skipping GPU {gpu_id} (no files assigned)")
    
    # Wait for all processes to complete
    print("\nWaiting for all processes to complete...")
    for p in processes:
        p.join()
        if p.exitcode == 0:
            print(f"Process {p.pid} finished successfully")
        else:
            print(f"Process {p.pid} finished with exit code {p.exitcode}")
    
    print("\nAll processes completed!")
