import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from typing import Dict, Optional
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import default_collate
from sam_3d_body.data.transforms import (
    Compose,
    TopdownAffine,
    VisionTransformWrapper,
)
from .bedlam.utils.image_utils import read_img

# def convert_bbox_centre_hw_to_corners(centre, height, width):
#     x1 = centre[0] - height/2.0
#     x2 = centre[0] + height/2.0
#     y1 = centre[1] - width/2.0
#     y2 = centre[1] + width/2.0

#     return np.array([x1, y1, x2, y2], dtype=np.int16)


class FakeGetBBoxCenterScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, results: Dict) -> Optional[dict]:
        results["bbox_center"] = results["center"].astype(np.float32)
        results["bbox_scale"] = (
            np.array([results["scale"], results["scale"]], dtype=np.float32) * 200.0
        )
        return results


class SSP3DDataset(Dataset):
    def __init__(self, ssp3d_dir_path, cfg):
        super(SSP3DDataset, self).__init__()
        self.cfg = cfg

        self.images_dir = os.path.join(ssp3d_dir_path, "images")
        self.silhouettes_dir = os.path.join(ssp3d_dir_path, "silhouettes")

        data = np.load(os.path.join(ssp3d_dir_path, "labels_mhr.npz"))

        self.image_fnames = data["fnames"]
        self.body_shapes = data["shapes"]
        self.body_poses = data["poses"]
        self.cam_trans = data["cam_trans"]
        self.joints2D = data["joints2D"]
        self.bbox_centres = data["bbox_centres"]  # Tight bounding box centre
        self.bbox_whs = data["bbox_whs"]  # Tight bounding box width/height
        self.genders = data["genders"]

        self.mhr_shape = data["identity_coeffs"]
        self.mhr_pose = data["lbs_model_params"]
        self.mhr_expr = data["face_expr_coeffs"]

        self.focal_length = 5000.0

        self.transform = Compose(
            [
                FakeGetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.DATASET.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        item = {}
        if torch.is_tensor(index):
            index = index.tolist()

        fname = self.image_fnames[index]
        image = read_img(os.path.join(self.images_dir, fname))
        silhouette = cv2.imread(os.path.join(self.silhouettes_dir, fname), 0)
        joints2D = self.joints2D[index]
        shape = self.body_shapes[index]
        pose = self.body_poses[index]
        cam_trans = self.cam_trans[index]
        gender = self.genders[index]
        mhr_shape = self.mhr_shape[index]
        mhr_pose = self.mhr_pose[index]
        mhr_expr = self.mhr_expr[index]

        # SSP-3D already cropped
        assert image.shape[0] == image.shape[1], "Assert ssp3d square image only"
        bbox_center = np.array(image.shape[:2]) / 2
        bbox_scale = image.shape[1] / 200.0

        # Build camera intrinsics matrix
        img_h, img_w = image.shape[:2]
        cam_int = np.eye(3, dtype=np.float32)
        cam_int[0, 0] = self.focal_length  # fx
        cam_int[1, 1] = self.focal_length  # fy
        cam_int[0, 2] = img_w / 2.0  # cx (principal point x)
        cam_int[1, 2] = img_h / 2.0  # cy (principal point y)

        data_info = dict(
            img=image,
            center=bbox_center,
            scale=bbox_scale,
            bbox_format="xyxy",
        )
        data_list = [self.transform(data_info)]
        data = default_collate(data_list)

        for key in data:
            item[key] = data[key]

        item["person_valid"] = torch.ones((1, 1))
        item["img_ori"] = image
        item["mask"] = silhouette
        item["shape_params"] = mhr_shape
        item["model_params"] = mhr_pose
        item["face_expr_coeffs"] = mhr_expr
        item["ori_img_size"] = image.shape[:2]
        item["cam_int"] = cam_int
        item["trans_cam"] = cam_trans
        item["dataset_name"] = "ssp3d"
        return item


class MultiSSP3DDataset(Dataset):
    """Multi-view evaluation dataset that groups SSP3D data by unique body shapes.

    This dataset groups SSP3D data by unique body shapes (serno) and selects
    num_view images corresponding to each serno for multi-view evaluation.
    """

    def __init__(self, ssp3d_dir_path, num_view=4, cfg=None):
        super(MultiSSP3DDataset, self).__init__()

        self.dataset = SSP3DDataset(ssp3d_dir_path, cfg)
        self.num_view = num_view

        self.serno = np.unique(self.dataset.body_shapes, axis=0)

        self._group_by_serno()

        # Filter sernos that have at least num_view images
        self.valid_sernos = [
            serno
            for serno, indices in self.serno_to_indices.items()
            if len(indices) >= num_view
        ]

        if len(self.valid_sernos) == 0:
            raise ValueError(f"No serno found with at least {num_view} images")

        print(f"Found {len(self.valid_sernos)} sernos with at least {num_view} views")
        print(f"Total samples: {len(self.valid_sernos)}")

    def _group_by_serno(self):
        """Group all data indices by their unique body shape (serno)."""
        self.serno_to_indices = {}

        for serno in self.serno:
            indices = np.where(
                np.all(
                    np.isclose(self.dataset.body_shapes, serno, rtol=1e-5, atol=1e-8),
                    axis=1,
                )
            )[0]

            genders = self.dataset.genders[indices]
            assert np.all(
                genders == genders[0]
            ), f"All genders in group should be the same, found: {np.unique(genders)}"

            serno_key = tuple(serno) if serno.ndim > 0 else serno.item()
            self.serno_to_indices[serno_key] = indices.tolist()

    def _load_single_view(self, index):
        item = {}
        if torch.is_tensor(index):
            index = index.tolist()

        fname = self.dataset.image_fnames[index]
        image = read_img(os.path.join(self.dataset.images_dir, fname))
        silhouette = cv2.imread(os.path.join(self.dataset.silhouettes_dir, fname), 0)
        mhr_shape = torch.from_numpy(self.dataset.mhr_shape[index])
        mhr_pose = torch.from_numpy(self.dataset.mhr_pose[index])
        mhr_expr = torch.from_numpy(self.dataset.mhr_expr[index])
        cam_trans = torch.from_numpy(self.dataset.cam_trans[index])

        # SSP-3D already cropped
        assert image.shape[0] == image.shape[1], "Assert ssp3d square image only"
        bbox_center = np.array(image.shape[:2]) / 2
        bbox_scale = image.shape[1] / 200.0

        # Build camera intrinsics matrix
        img_h, img_w = image.shape[:2]
        cam_int = np.eye(3, dtype=np.float32)
        cam_int[0, 0] = self.dataset.focal_length  # fx
        cam_int[1, 1] = self.dataset.focal_length  # fy
        cam_int[0, 2] = img_w / 2.0  # cx (principal point x)
        cam_int[1, 2] = img_h / 2.0  # cy (principal point y)
        cam_int = torch.from_numpy(cam_int).float()

        data_info = dict(
            img=image,
            center=bbox_center,
            scale=bbox_scale,
            bbox_format="xyxy",
            mask=silhouette,
        )
        data_list = [self.dataset.transform(data_info)]
        data = default_collate(data_list)

        for key in data:
            item[key] = data[key]

        item["person_valid"] = torch.ones((1, 1))
        item["img_ori"] = image
        item["shape_params"] = mhr_shape
        item["model_params"] = mhr_pose
        item["face_expr_coeffs"] = mhr_expr
        item["cam_int"] = cam_int
        item["trans_cam"] = cam_trans

        return item

    def __getitem__(self, index):

        serno_idx = index % len(self.valid_sernos)
        selected_serno = self.valid_sernos[serno_idx]
        indices = self.serno_to_indices[selected_serno]

        selected_indices = np.random.choice(indices, size=self.num_view, replace=False)

        views = []
        for idx in selected_indices:
            views.append(self._load_single_view(int(idx)))

        item = {}

        for field in set(views[0].keys()):
            try:
                stacked = torch.stack([v[field] for v in views], dim=0)
                item[field] = stacked
            except:

                item[field] = [v[field] for v in views]

        item["mask"] = item["mask"].float().unsqueeze(-3)
        item["mask_score"] = torch.ones((len(views), 1, 1, 1))
        item["num_views"] = len(views)
        item["selected_serno"] = selected_serno
        item["selected_indices"] = selected_indices.tolist()
        item["dataset_name"] = "ssp3d"

        return item

    def __len__(self):
        """Return number of unique sernos with at least num_view images."""
        return len(self.valid_sernos)
