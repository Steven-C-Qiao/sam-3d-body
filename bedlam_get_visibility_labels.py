import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.renderer import PerspectiveCameras, TexturesVertex
from pytorch3d.structures import Meshes

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

    def forward(self, mesh, **kwargs):

        # images = self.renderer(mesh)

        fragments = self.renderer.rasterizer(mesh)

        ret = {
            # "maps": images[..., :-1],  # All channels except the last (alpha)
            # "mask": images[..., -1],  # Last channel is the alpha/mask
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


def check_joint_visibility(
    j2d,
    j2d_depth,
    fragments,
    depth_tolerance=0.01,
):
    """
    Args:
        j2d: (N, J, 2)
        j2d_depth: (N, J)
        fragments: (N, H, W, K)
        depth_tolerance: float
    Returns:
        visibility: (N, J)
    """
    zbuf = fragments.zbuf  # (N, H, W, K)
    zbuf = torch.flip(zbuf, dims=[1, 2])  # NOTE: Flip here similar to the image
    N, H, W, K = zbuf.shape

    # j2d: (N, J, 2)
    x_pixel, y_pixel = j2d[..., 0], j2d[..., 1]  # (N, J)

    # Round to integer pixel coordinates for indexing
    x_idx = torch.clamp(torch.round(x_pixel).long(), 0, W - 1)
    y_idx = torch.clamp(torch.round(y_pixel).long(), 0, H - 1)

    in_bounds = (x_pixel >= 0) & (x_pixel < W) & (y_pixel >= 0) & (y_pixel < H)



    # Gather depths at each (N, J) pixel location for all faces_per_pixel K.
    # Build batch indices with same shape as (N, J) so advanced indexing
    # produces an output of shape (N, J, K), not (N, N, J, K).
    batch_idx = torch.arange(N, device=zbuf.device)[:, None].expand(-1, y_idx.shape[1])  # (N, J)
    depths_at_pixels = zbuf[batch_idx, y_idx, x_idx, :]  # (N, J, K)

    valid_mask = depths_at_pixels > -1


    # Count how many faces are strictly in front of the joint along the ray
    # j2d_depth: (N, J) -> (N, J, 1) to compare with (N, J, K)
    front_faces = depths_at_pixels < (
        j2d_depth.unsqueeze(-1) - depth_tolerance
    )  # (N, J, K)
    front_faces = front_faces & valid_mask
    num_front_faces = front_faces.sum(dim=-1)  # (N, J)

    any_valid = valid_mask.any(dim=-1)  # (N, J)

    visibility = in_bounds & any_valid & (num_front_faces <= 1)
    return visibility


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mhr_model = torch.jit.load(
        MHR_MODEL_PATH,
        map_location=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ckpt = torch.load(
        "/scratch/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt",
        weights_only=False,
    )
    faces = ckpt["head_pose.faces"].cpu().detach().numpy()
    faces = torch.tensor(faces, dtype=torch.long, device=device)


    all_npzs = sorted([
        os.path.join(DATA_BASE_PATH, "training_labels/all_npz_12_training_extra_mhr", f)
        for f in os.listdir(
            os.path.join(
                DATA_BASE_PATH, "training_labels/all_npz_12_training_extra_mhr"
            )
        )
        if (f.endswith(".npz") and not f.endswith("_visibility.npz"))
    ])

    
    from tqdm import tqdm

    with torch.no_grad():
        for npz_path in all_npzs:
            path_if_already_done = npz_path[:-4] + "_visibility.npz"
            if os.path.exists(path_if_already_done):
                print(f'{npz_path} already done')
                continue


            vertices, joints3d = [], []
            data = np.load(npz_path)

            new_data = {}
            for key in data.keys():
                try:
                    new_data[key] = torch.tensor(data[key], dtype=torch.float32, device=device)
                except:
                    print(f'{key} not converted to tensor')

            data = new_data

            chunk_size = 64
            num_samples = data["identity_coeffs"].shape[0]
            for i in tqdm(range(0, num_samples, chunk_size)):
                end_idx = min(i + chunk_size, num_samples)

                verts, skeleton = mhr_model(
                    data["identity_coeffs"][i:end_idx],
                    data["lbs_model_params"][i:end_idx],
                    data["face_expr_coeffs"][i:end_idx],
                )
                verts /= 100.0
                j3d = skeleton[:, :, :3] / 100.0
                vertices.append(verts)
                joints3d.append(j3d)

            vertices = torch.cat(vertices, dim=0)
            joints3d = torch.cat(joints3d, dim=0)


            closeup = "closeup" in npz_path.lower()
            image_size = (720, 1280) if not closeup else (1280, 720)

            downscale_by = 2
            image_size = (int(image_size[0] / downscale_by), int(image_size[1] / downscale_by))
            
            
            renderer = FeatureRenderer(
                image_size=image_size
            )

            all_visibilities = []
            for i in tqdm(range(0, num_samples, chunk_size)):
                
                end_idx = min(i + chunk_size, num_samples)

                verts = vertices[i:end_idx]
                j3d = joints3d[i:end_idx]

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


                j2d, j2d_depth = project(
                    j3d,
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
                # pytorch3d_output = pytorch3d_output["maps"].cpu().detach().numpy()
                # pytorch3d_output = pytorch3d_output[:, ::-1, ::-1, :]

                visibility = check_joint_visibility(
                    j2d,
                    j2d_depth,
                    fragments=fragments,
                    depth_tolerance=0.01,  # Adjust based on your scale
                ).cpu().numpy()

                all_visibilities.append(visibility)


                # fig = plt.figure()
                # axes = fig.add_subplot(1, 1, 1)
                # axes.imshow(pytorch3d_output[0])
                # axes.set_title("PyTorch3D Renderer")
                # scatter = axes.scatter(
                #     j2d[0, :, 0].cpu().detach().numpy(), j2d[0, :, 1].cpu().detach().numpy(), c=visibility[0], cmap="cool", s=1, vmin=0, vmax=1
                # )
                # plt.tight_layout()
                # plt.savefig('visibility.png')
                # plt.close()

            all_visibilities = np.concatenate(all_visibilities, axis=0)
            np.savez(os.path.join(npz_path[:-4] + "_visibility.npz"), visibility=all_visibilities)
            print(f'Saved visibility labels to {npz_path[:-4] + "_visibility.npz"}')
