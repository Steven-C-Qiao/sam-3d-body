import cv2
import os
import torch
import pickle
import numpy as np
from loguru import logger
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import default_collate


from .bedlam import constants
from .bedlam.constants import NUM_JOINTS_SMPLX
from .bedlam.utils.image_utils import random_crop, read_img
from ..configs.config import DATASET_FILES, DATASET_FOLDERS
from ..configs.config import (
    SMPL_MODEL_DIR,
    SMPLX_MODEL_DIR,
    JOINT_REGRESSOR_H36M,
    SMPLX2SMPL,
)
from smplx import SMPL, SMPLX


from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)


import torch.nn as nn
from typing import Dict, Optional


class FakeGetBBoxCenterScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, results: Dict) -> Optional[dict]:
        results["bbox_center"] = results["center"].astype(np.float32)
        results["bbox_scale"] = (
            np.array([results["scale"], results["scale"]], dtype=np.float32) * 200.0
        )
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


class DatasetHMR(Dataset):

    def __init__(self, options, dataset, use_augmentation=True, is_train=True):
        super(DatasetHMR, self).__init__()
        self.dataset = dataset
        self.is_train = True
        self.options = options
        self.use_augmentation = use_augmentation
        self.img_dir = DATASET_FOLDERS[dataset]
        self.mask_dir = (
            self.img_dir.replace("training_images", "masks")
            .replace("_6fps/", "/")
            .replace("_30fps/", "/")
            .replace("png", "masks")
        )
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.visibility = np.load(DATASET_FILES[is_train][dataset][:-4] + "_visibility.npz")["visibility"]
        self.imgname = self.data["imgname"]
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data["scale"].astype(np.float32)
        self.center = self.data["center"].astype(np.float32)

        input_size = options.IMAGE_SIZE
        self.transform = Compose(
            [
                FakeGetBBoxCenterScale(),
                TopdownAffine(input_size=input_size, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.cam_int = self.data["cam_int"].astype(np.float32)
        self.shape_params = self.data["identity_coeffs"]
        self.model_params = self.data["lbs_model_params"]
        self.face_expr_params = self.data["face_expr_coeffs"]

        self.cam_ext = self.data["cam_ext"]
        self.trans_cam = self.data["trans_cam"]

        # self.mhr_keypoints_2d = self.data["mhr_keypoints_2d"]

        # evaluation variables
        # if not self.is_train:
        #     if "width" in self.data:  # For closeup image stored in rotated format
        #         self.width = self.data["width"]
        #     self.joint_mapper_h36m = constants.H36M_TO_J14
        #     self.joint_mapper_gt = constants.J24_TO_J14
        #     self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        #     self.smpl_male = SMPL(SMPL_MODEL_DIR, gender="male", create_transl=False)
        #     self.smpl_female = SMPL(
        #         SMPL_MODEL_DIR, gender="female", create_transl=False
        #     )
        #     self.smplx_male = SMPLX(SMPLX_MODEL_DIR, gender="male")
        #     self.smplx_female = SMPLX(SMPLX_MODEL_DIR, gender="female")
        #     self.smplx2smpl = pickle.load(open(SMPLX2SMPL, "rb"))
        #     self.smplx2smpl = torch.tensor(
        #         self.smplx2smpl["matrix"][None], dtype=torch.float32
        #     )
        # if (
        #     self.is_train and "agora" not in self.dataset and "3dpw" not in self.dataset
        # ):  # first 80% is training set 20% is validation
        #     self.length = int(self.scale.shape[0] * self.options.CROP_PERCENT)
        # else:
        self.length = self.scale.shape[0]
        logger.info(f"Loaded {self.dataset} dataset, num samples {self.length}")

    def scale_aug(self):
        sc = 1  # scaling
        if self.is_train:
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + self.options.SCALE_FACTOR,
                max(
                    1 - self.options.SCALE_FACTOR,
                    np.random.randn() * self.options.SCALE_FACTOR + 1,
                ),
            )
        return sc

    def rgb_processing(self, rgb_img_full):
        """Apply albumentation augmentations to the image."""
        if self.is_train and self.options.ALB:
            aug_comp = [
                A.Downscale((0.5, 0.9), p=0.1),
                # A.ImageCompression((20, 100), p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.Blur(blur_limit=(3, 11), p=0.1),
                A.RandomSnow(brightness_coeff=1.5, snow_point_range=(0.2, 0.4)),
            ]
            aug_mod = [
                A.CLAHE((1, 11), (10, 10), p=0.2),
                A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(
                    multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.2,
                ),
                A.Posterize(p=0.1),
                A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                A.Equalize(mode="cv", p=0.1),
            ]
            albumentation_aug = A.Compose(
                [
                    A.OneOf(aug_comp, p=self.options.ALB_PROB),
                    A.OneOf(aug_mod, p=self.options.ALB_PROB),
                ]
            )
            rgb_img_full = albumentation_aug(image=rgb_img_full)["image"]
        return rgb_img_full

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        mhr_keypoints_2d = self.mhr_keypoints_2d[index].copy()

        sc = 1.0

        imgname = os.path.join(self.img_dir, self.imgname[index])
        maskname = os.path.join(self.mask_dir, self.imgname[index][:-4] + "_env.png")
        item["imgname"] = imgname
    
        cv_img = read_img(imgname)
        masks = cv2.imread(maskname, 0)
        if "closeup" in self.dataset:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            masks = cv2.rotate(masks, cv2.ROTATE_90_CLOCKWISE)

        item["img_ori"] = cv_img
        item["mask_ori"] = masks

        img = self.rgb_processing(cv_img)

        data_info = dict(
            img=img,
            center=center,
            scale=float(sc * scale),
            bbox_format="xyxy",
            keypoints_2d=mhr_keypoints_2d,
            mask=masks,
        )
        data_list = [self.transform(data_info)]
        data = default_collate(data_list)

        for key in data:
            item[key] = data[key]
        item["mask"] = item["mask"].float().unsqueeze(-3) * -1.0  # N, 1, H, W
        item["mask_score"] = torch.ones((item["mask"].shape[0], 1, 1, 1))
    
        item["person_valid"] = torch.ones((1, 1))
        
        item["scale"] = sc * scale
        item["center"] = center.astype(np.float32)
        item["shape_params"] = self.shape_params[index]
        item["model_params"] = self.model_params[index]
        item["face_expr_coeffs"] = self.face_expr_params[index]
        item["scale_params"] = self.model_params[index, -68:]
        item["visibility"] = self.visibility[index]

        item["cam_int"] = self.cam_int[index]
        item["focal_length"] = torch.tensor(
            [self.cam_int[index][0, 0], self.cam_int[index][1, 1]]
        )
        item["cam_ext"] = self.cam_ext[index] + self.trans_cam[index]
        item["trans_cam"] = self.trans_cam[index]

        item["dataset_name"] = self.dataset

        # if not self.is_train:
        #     item["dataset_index"] = self.options.VAL_DS.split("_").index(self.dataset)

        return item

    def __len__(self):
        if self.is_train and "agora" not in self.dataset and "3dpw" not in self.dataset:
            return int(self.options.CROP_PERCENT * len(self.imgname))
        else:
            return len(self.imgname)


class MultiViewEvaluationDataset(Dataset):
    """Evaluation dataset that loads multiple views for uncertainty merging.

    This dataset groups BEDLAM data by serial number ('serno') and randomly
    selects one serno per __getitem__ call, then loads num_view images
    corresponding to that serno.
    """

    def __init__(
        self, options, dataset, num_view=4, use_augmentation=False, is_train=False
    ):
        super(MultiViewEvaluationDataset, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.num_view = num_view
        self.img_dir = DATASET_FOLDERS[dataset]
        self.mask_dir = (
            self.img_dir.replace("training_images", "masks")
            .replace("_6fps/", "/")
            .replace("png", "masks")
        )
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.imgname = self.data["imgname"]
        self.scale = self.data["scale"].astype(np.float32)
        self.center = self.data["center"].astype(np.float32)
        self.cam_int = self.data["cam_int"].astype(np.float32)
        self.shape_params = self.data["identity_coeffs"].astype(np.float32)
        self.model_params = self.data["lbs_model_params"].astype(np.float32)
        self.face_expr_params = self.data["face_expr_coeffs"].astype(np.float32)
        self.cam_ext = self.data["cam_ext"].astype(np.float32)
        self.trans_cam = self.data["trans_cam"].astype(np.float32)

        # Load serno field
        if "serno" not in self.data.files:
            raise KeyError(
                "'serno' field not found in dataset. This dataset requires serial numbers to group views."
            )
        self.serno = self.data["serno"]

        self.pose_cam = self.data["pose_cam"][:, : NUM_JOINTS_SMPLX * 3].astype(np.float32)
        self.betas = self.data["shape"].astype(np.float32)

        # Load keypoints
        full_joints = self.data["gtkps"]
        self.keypoints = full_joints[:, :24]

        self.mhr_keypoints_2d = self.data["mhr_keypoints_2d"]

        # Setup transforms (no augmentation for evaluation)
        input_size = self.options.IMAGE_SIZE
        self.transform = Compose(
            [
                FakeGetBBoxCenterScale(),
                TopdownAffine(input_size=input_size, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

        # Group indices by serno
        self._group_by_serno()

        # Filter sernos that have at least num_view images
        self.valid_sernos = [
            serno
            for serno, indices in self.serno_to_indices.items()
            if len(indices) >= num_view
        ]

        if len(self.valid_sernos) == 0:
            raise ValueError(f"No serno found with at least {num_view} images")

        logger.info(f"Loaded {self.dataset} multi-view dataset")
        logger.info(
            f"Found {len(self.valid_sernos)} sernos with at least {num_view} views"
        )
        logger.info(f"Total samples: {len(self.valid_sernos)}")

    def _group_by_serno(self):
        """Group all data indices by their serial number."""
        self.serno_to_indices = {}
        for idx in range(len(self.imgname)):
            serno = self.serno[idx]

            serno = tuple(serno) if serno.ndim > 0 else serno.item()
            if serno not in self.serno_to_indices:
                self.serno_to_indices[serno] = []
            self.serno_to_indices[serno].append(idx)

    def _select_diverse_viewpoints(self, indices, num_view):
        """
        Select num_view indices with somewhat diverse viewpoints.
        Uses a randomized greedy approach to ensure diversity while allowing variation.

        Args:
            indices: List of indices for a given serno
            num_view: Number of views to select

        Returns:
            Selected indices with diverse viewpoints
        """
        if len(indices) <= num_view:
            return np.array(indices)

        # Extract rotation angles (viewpoints) for all indices
        # lbs_model_params[:, 3:6] contains global rotation (Euler angles)
        viewpoints = self.model_params[indices, 3:6]  # [N, 3] Euler angles

        # Compute pairwise angular distances between rotations
        n = len(indices)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Compute angular distance between two Euler angle rotations
                diff = viewpoints[i] - viewpoints[j]
                # Handle angle wrapping (Euler angles are periodic)
                diff = np.abs(diff)
                diff = np.minimum(diff, 2 * np.pi - diff)
                dist = np.linalg.norm(diff)
                distances[i, j] = dist
                distances[j, i] = dist

        # Randomized greedy algorithm: select views with diversity but allow variation
        selected = []
        remaining = set(range(n))

        # Start with a random view (adds variation)
        first_idx = np.random.choice(list(remaining))
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Greedily add views that are diverse from already selected views
        # But allow some randomness by considering top-k candidates
        while len(selected) < num_view and remaining:
            candidates_with_distances = []
            for candidate in remaining:
                # Minimum distance from candidate to any selected view
                min_dist_to_selected = min(
                    distances[candidate, sel] for sel in selected
                )
                candidates_with_distances.append((candidate, min_dist_to_selected))

            # Sort by distance (descending) and consider top candidates
            candidates_with_distances.sort(key=lambda x: x[1], reverse=True)

            # Select from top 3 candidates (or all if fewer than 3) to add randomness
            top_k = min(3, len(candidates_with_distances))
            if top_k > 0:
                top_candidates = [c[0] for c in candidates_with_distances[:top_k]]
                chosen_idx = np.random.choice(top_candidates)
                selected.append(chosen_idx)
                remaining.remove(chosen_idx)
            else:
                break

        # Convert back to original indices
        selected_indices = np.array([indices[i] for i in selected])
        return selected_indices

    def _load_single_view(self, index):
        """Load a single view (similar to DatasetHMR.__getitem__ but returns dict)."""
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        mhr_keypoints_2d = self.mhr_keypoints_2d[index].copy()
        mhr_keypoints_2d_orig = self.mhr_keypoints_2d[index].copy()

        sc = 1.0  # No augmentation for evaluation

        imgname = os.path.join(self.img_dir, self.imgname[index])
        maskname = os.path.join(self.mask_dir, self.imgname[index][:-4] + "_env.png")
        try:
            cv_img = read_img(imgname)
            masks = cv2.imread(maskname, 0)
        except Exception as E:
            print(E)
            logger.info(f"@{imgname}@ from {self.dataset}")
            # Return None to indicate failure
            return None

        item["img_ori"] = cv_img
        item["mask_ori"] = masks
        if "closeup" in self.dataset:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            masks = cv2.rotate(masks, cv2.ROTATE_90_CLOCKWISE)

        orig_shape = np.array(cv_img.shape)[:2]
        pose = self.pose_cam[index].copy()
        img = cv_img

        # Prepare data for transform pipeline
        data_info = dict(
            img=img,
            center=center,
            scale=float(sc * scale),
            bbox_format="xyxy",
            keypoints_2d=mhr_keypoints_2d,
            mask=masks,
        )

        # Apply transform pipeline
        data_list = [self.transform(data_info)]
        data = default_collate(data_list)

        # Extract transformed image and keypoints
        img = data["img"]
        transformed_keypoints_2d = data["keypoints_2d"]

        # Normalize keypoints to [-1, 1]
        topdown_affine = self.transform.transforms[1]
        input_size = topdown_affine.input_size
        if isinstance(input_size, tuple):
            img_w, img_h = input_size
        else:
            img_w = img_h = input_size

        normalized_keypoints = transformed_keypoints_2d.clone()
        normalized_keypoints[:, 0] = 2.0 * normalized_keypoints[:, 0] / img_w - 1.0
        normalized_keypoints[:, 1] = 2.0 * normalized_keypoints[:, 1] / img_h - 1.0

        keypoints_vis = (
            keypoints[:, 2:3]
            if keypoints.shape[1] > 2
            else np.ones((keypoints.shape[0], 1))
        )
        if normalized_keypoints.shape[1] == 2:
            normalized_keypoints = np.hstack(
                [normalized_keypoints.cpu().numpy(), keypoints_vis]
            )
            normalized_keypoints = torch.from_numpy(normalized_keypoints).float()

        item["person_valid"] = torch.ones((1, 1))

        # Store all transform outputs
        for key in data:
            item[key] = data[key]

        item["mask"] = item["mask"].float().unsqueeze(-3) * -1.0  # N, 1, H, W
        item["mask_score"] = torch.ones((item["mask"].shape[0], 1, 1, 1))
        item["cam_int"] = torch.from_numpy(self.cam_int[index]).float()
        item["shape_params"] = torch.from_numpy(self.shape_params[index]).float()
        item["model_params"] = torch.from_numpy(self.model_params[index]).float()
        item["face_expr_coeffs"] = torch.from_numpy(
            self.face_expr_params[index]
        ).float()
        item["scale_params"] = torch.from_numpy(self.model_params[index, -68:]).float()

        item["img"] = img
        item["pose"] = torch.from_numpy(pose).float()
        item["betas"] = torch.from_numpy(self.betas[index]).float()
        item["imgname"] = imgname
        if "cam_int" in self.data.files:
            item["focal_length"] = torch.tensor(
                [self.cam_int[index][0, 0], self.cam_int[index][1, 1]]
            )
        item["cam_ext"] = self.cam_ext[index]
        item["translation"] = self.cam_ext[index][:, 3]
        if "trans_cam" in self.data.files:
            # NOTE: This also modifies 'trans_cam', which results in the correct cam_t
            item["translation"][:3] += self.trans_cam[index]

        item["trans_cam"] = torch.from_numpy(self.trans_cam[index]).float()
        item["cam_ext"] = torch.from_numpy(self.cam_ext[index]).float()
        item["keypoints_orig"] = torch.from_numpy(mhr_keypoints_2d_orig).float()
        item["keypoints"] = normalized_keypoints.float()
        item["scale"] = float(sc * scale)
        item["center"] = center.astype(np.float32)
        item["orig_shape"] = orig_shape
        item["sample_index"] = index
        item["dataset_name"] = self.dataset
        item["serno"] = self.serno[index]

        return item

    def __getitem__(self, index):
        """Load num_view images from a randomly selected serno.

        Args:
            index: Dataset index (used to select which serno to use)

        Returns:
            Dictionary with multi-view data. Each field contains a list of tensors
            for each view, or stacked tensors where appropriate.
        """
        # Select serno (use index modulo to cycle through valid sernos)
        serno_idx = index % len(self.valid_sernos)
        selected_serno = self.valid_sernos[serno_idx]
        indices = self.serno_to_indices[selected_serno]

        # Select diverse viewpoints instead of random sampling
        if len(indices) > self.num_view:
            selected_indices = self._select_diverse_viewpoints(indices, self.num_view)
        else:
            # If we have fewer than num_view, use all available and pad if needed
            selected_indices = np.array(indices.copy())
            if len(selected_indices) < self.num_view:
                # Pad by repeating the last index
                selected_indices = np.concatenate(
                    [
                        selected_indices,
                        np.repeat(
                            selected_indices[-1:],
                            self.num_view - len(selected_indices),
                            axis=0,
                        ),
                    ]
                )

        # Load all views
        views = []
        for idx in selected_indices:
            view_data = self._load_single_view(idx)
            if view_data is None:
                # If loading failed, skip this view (or use a placeholder)
                continue
            views.append(view_data)

        if len(views) == 0:
            raise RuntimeError(f"Failed to load any views for serno {selected_serno}")

        # If we have fewer views than num_view, pad with the last view
        while len(views) < self.num_view:
            views.append(views[-1].copy() if views else None)

        # Stack or list the views appropriately
        # Try stacking for all fields; if it fails, keep as list
        multi_view_item = {}

        # Get all fields from the first view
        all_fields = set(views[0].keys())

        for field in all_fields:
            if field in views[0]:
                try:
                    # Try to stack tensors
                    stacked = torch.stack([v[field] for v in views], dim=0)
                    multi_view_item[field] = stacked
                except:
                    # If stacking fails, keep as list
                    multi_view_item[field] = [v[field] for v in views]

        # Add metadata
        multi_view_item["num_views"] = len(views)
        multi_view_item["selected_serno"] = selected_serno
        multi_view_item["selected_indices"] = selected_indices.tolist()

        return multi_view_item

    def __len__(self):
        """Return number of unique sernos with at least num_view images."""
        return len(self.valid_sernos)


if __name__ == "__main__":

    def _test_multiview_dataset():
        """
        Simple sanity check for MultiViewEvaluationDataset.

        Adjust `dataset_name` below to one of the BEDLAM dataset keys you have
        available in `DATASET_FILES[is_train]` (see bedlam/config.py).
        """
        from torch.utils.data import DataLoader

        class DummyOptions:
            # Only fields that might be accessed by downstream code
            SCALE_FACTOR = 0.0
            ALB = False
            ALB_PROB = 0.0
            CROP_PERCENT = 1.0
            CROP_FACTOR = 0.0
            CROP_PROB = 0.0
            VAL_DS = ""
            DATASETS_AND_RATIOS = "static-hdri"

        options = DummyOptions()
        # TODO: change this to a dataset that exists in your setup
        dataset_name = "static-hdri"

        print(f"Creating MultiViewEvaluationDataset for '{dataset_name}'...")
        ds = MultiViewEvaluationDataset(
            options=options,
            dataset=dataset_name,
            num_view=4,
            is_train=True,  # doesn't really matter here
        )

        print(f"Number of unique sernos with at least 4 views: {len(ds)}")

        # Create DataLoader
        batch_size = 2
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for debugging, increase for faster loading
            pin_memory=False,
        )

        print(f"\nLoading batch with batch_size={batch_size}...")
        batch = next(iter(dataloader))

        print("\nBatch keys:", batch.keys())
        print(f"Batch size (number of sernos): {len(batch.get('selected_serno', []))}")

        if "img" in batch:
            print(f"img shape (batch, views, C, H, W): {batch['img'].shape}")
        if "keypoints" in batch:
            print(f"keypoints shape (batch, views, N, 3): {batch['keypoints'].shape}")
        if "pose" in batch:
            print(f"pose shape (batch, views, ...): {batch['pose'].shape}")
        if "betas" in batch:
            print(f"betas shape (batch, views, ...): {batch['betas'].shape}")
        if "selected_serno" in batch:
            print(f"Selected sernos: {batch['selected_serno']}")
        if "num_views" in batch:
            print(f"Number of views per sample: {batch['num_views']}")

        import ipdb

        ipdb.set_trace()
        print("")

    _test_multiview_dataset()
