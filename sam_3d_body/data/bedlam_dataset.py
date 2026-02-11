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
        self.img_dir = DATASET_FOLDERS[dataset]
        # Clean up directory paths of any sub-strings '_6fps' or '_30fps'
        self.mask_dir = (
            self.img_dir.replace('training_images', 'masks')
            .replace('_6fps/', '/')
            .replace('_30fps/', '/')
            .replace('png', 'masks')
        )
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.imgname = self.data["imgname"]
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data["scale"].astype(np.float32)
        self.center = self.data["center"].astype(np.float32)

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        input_size = (256, 256)
        self.transform = Compose(
            [
                # GetBBoxCenterScale(),
                FakeGetBBoxCenterScale(),
                TopdownAffine(input_size=input_size, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.cam_int = self.data["cam_int"].astype(np.float32)
        self.shape_params = self.data["identity_coeffs"].astype(np.float32)
        self.model_params = self.data["lbs_model_params"].astype(np.float32)
        self.face_expr_params = self.data["face_expr_coeffs"].astype(np.float32)

        self.cam_ext = self.data["cam_ext"].astype(np.float32)
        self.trans_cam = self.data["trans_cam"].astype(np.float32)

        if self.is_train:
            if "3dpw-train-smplx" in self.dataset:
                self.pose_cam = self.data["smplx_pose"][
                    :, : NUM_JOINTS_SMPLX * 3
                ].astype(float)
                self.betas = self.data["smplx_shape"][:, :11].astype(float)
            else:
                self.pose_cam = self.data["pose_cam"][:, : NUM_JOINTS_SMPLX * 3].astype(
                    float
                )
                self.betas = self.data["shape"].astype(float)

            # For AGORA and 3DPW num betas are 10
            if self.betas.shape[-1] == 10:
                self.betas = np.hstack((self.betas, np.zeros((self.betas.shape[0], 1))))

            # if 'cam_int' in self.data:
            #     self.cam_int = self.data['cam_int']
            # else:
            #     self.cam_int = np.zeros((self.imgname.shape[0], 3, 3))
            if "cam_ext" in self.data:
                self.cam_ext = self.data["cam_ext"]
            else:
                self.cam_ext = np.zeros((self.imgname.shape[0], 4, 4))
            if "trans_cam" in self.data:
                self.trans_cam = self.data["trans_cam"]

        else:
            if (
                "h36m" in self.dataset
            ):  # H36m doesn't have pose and shape param only 3d joints
                self.joints = self.data["S"]
                self.pose_cam = np.zeros((self.imgname.shape[0], 66))
                self.betas = np.zeros((self.imgname.shape[0], 11))
            else:
                self.pose_cam = self.data["pose_cam"].astype(float)
                self.betas = self.data["shape"].astype(float)

        if self.is_train:
            if "3dpw-train-smplx" in self.dataset:  # Only for 3dpw training
                self.joint_map = constants.joint_mapping(
                    constants.COCO_18, constants.SMPL_24
                )
                self.keypoints = np.zeros((len(self.imgname), 24, 3))
                self.keypoints = self.data["gtkps"][:, self.joint_map]
                self.keypoints[:, self.joint_map == -1] = -2
            else:
                full_joints = self.data["gtkps"]
                self.keypoints = full_joints[:, :24]
        else:
            self.keypoints = np.zeros((len(self.imgname), 24, 3))

        self.mhr_keypoints_2d = self.data["mhr_keypoints_2d"]

        if "proj_verts" in self.data:
            self.proj_verts = self.data["proj_verts"]
        else:
            self.proj_verts = np.zeros((len(self.imgname), 437, 3))

        try:
            gender = self.data["gender"]
            self.gender = np.array([0 if str(g) == "m" else 1 for g in gender]).astype(
                np.int32
            )
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        # evaluation variables
        if not self.is_train:
            if "width" in self.data:  # For closeup image stored in rotated format
                self.width = self.data["width"]
            self.joint_mapper_h36m = constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
            self.smpl_male = SMPL(SMPL_MODEL_DIR, gender="male", create_transl=False)
            self.smpl_female = SMPL(
                SMPL_MODEL_DIR, gender="female", create_transl=False
            )
            self.smplx_male = SMPLX(SMPLX_MODEL_DIR, gender="male")
            self.smplx_female = SMPLX(SMPLX_MODEL_DIR, gender="female")
            self.smplx2smpl = pickle.load(open(SMPLX2SMPL, "rb"))
            self.smplx2smpl = torch.tensor(
                self.smplx2smpl["matrix"][None], dtype=torch.float32
            )
        if (
            self.is_train and "agora" not in self.dataset and "3dpw" not in self.dataset
        ):  # first 80% is training set 20% is validation
            self.length = int(self.scale.shape[0] * self.options.CROP_PERCENT)
        else:
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
                A.RandomSnow(
                    brightness_coeff=1.5, snow_point_range=(0.2, 0.4)
                ),
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
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()
        mhr_keypoints_2d = self.mhr_keypoints_2d[index].copy()
        mhr_keypoints_2d_orig = self.mhr_keypoints_2d[index].copy()

        # if self.options.proj_verts:
        #     proj_verts_orig = self.proj_verts[index].copy()
        #     item['proj_verts_orig'] = torch.from_numpy(proj_verts_orig).float()
        #     proj_verts = self.proj_verts[index].copy()

        # Apply scale augmentation
        # sc = self.scale_aug()
        sc = 1.0
        # apply crop augmentation
        # if self.is_train and self.options.CROP_FACTOR > 0:
        #     rand_no = np.random.rand()
        #     if rand_no < self.options.CROP_PROB:
        #         center, scale = random_crop(center, scale,
        #                                     crop_scale_factor=1-self.options.CROP_FACTOR,
        #                                     axis='y')

        imgname = os.path.join(self.img_dir, self.imgname[index])
        maskname = os.path.join(self.mask_dir, self.imgname[index][:-4] + "_env.png")
        cv_img = read_img(imgname)

        # Robustly handle missing/invalid mask files: cv2.imread returns None on failure,
        # but only emits a warning. Explicitly check for None instead of relying on assert.
        masks = cv2.imread(maskname, 0)
        if masks is None:
            print(f"@{maskname}@ not found")
            masks = np.ones_like(cv_img)
        item["img_ori"] = cv_img
        item["mask_ori"] = masks
        if "closeup" in self.dataset:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            masks = cv2.rotate(masks, cv2.ROTATE_90_CLOCKWISE)

        orig_shape = np.array(cv_img.shape)[:2]
        pose = self.pose_cam[index].copy()

        # Apply albumentation augmentations
        # try:
        img = self.rgb_processing(cv_img)
        # img = cv_img
        # except Exception as E:
        #     logger.info(f'@{imgname} from {self.dataset}')
        #     print(E)
        #     img = cv_img

        # Prepare data for sam_3d_body transform pipeline
        # Keypoints should be in image coordinates (x, y, visibility)
        # Only pass visible keypoints (visibility > 0)

        # These are SMPL-X keypoints
        keypoints_2d = keypoints[:, :2].copy()  # Extract x, y coordinates
        keypoints_vis = (
            keypoints[:, 2:3]
            if keypoints.shape[1] > 2
            else np.ones((keypoints.shape[0], 1))
        )

        data_info = dict(
            img=img,
            center=center,
            scale=float(sc * scale),
            bbox_format="xyxy",
            keypoints_2d=mhr_keypoints_2d,
            mask=masks
        )
        # if self.options.proj_verts:
        #     proj_verts_2d = proj_verts[:, :2].copy()
        #     data_info["proj_verts_2d"] = proj_verts_2d

        # Apply sam_3d_body transform pipeline (crops image and transforms keypoints)
        data_list = [self.transform(data_info)]
        data = default_collate(data_list)

        # Extract transformed image and keypoints
        img = data["img"]  # Already a tensor from VisionTransformWrapper(ToTensor())
        transformed_keypoints_2d = data["keypoints_2d"]

        # Convert keypoints to normalized coordinates [-1, 1]
        # Keypoints are now in the transformed image space (input_size)
        # Get input_size from the transform pipeline (TopdownAffine is at index 1)
        topdown_affine = self.transform.transforms[1]
        input_size = topdown_affine.input_size
        if isinstance(input_size, tuple):
            img_w, img_h = input_size
        else:
            img_w = img_h = input_size

        # Normalize keypoints to [-1, 1]
        normalized_keypoints = transformed_keypoints_2d.clone()
        normalized_keypoints[:, 0] = 2.0 * normalized_keypoints[:, 0] / img_w - 1.0
        normalized_keypoints[:, 1] = 2.0 * normalized_keypoints[:, 1] / img_h - 1.0

        # Combine with visibility
        if normalized_keypoints.shape[1] == 2:
            normalized_keypoints = np.hstack([normalized_keypoints, keypoints_vis])

        # # Handle proj_verts if needed
        # # TopdownAffine doesn't automatically transform proj_verts_2d, so we need to do it manually
        # if self.options.proj_verts:
        #     if "affine_trans" in data:
        #         # Transform proj_verts using the same affine transformation
        #         warp_mat = data["affine_trans"]
        #         transformed_proj_verts_2d = cv2.transform(
        #             proj_verts_2d[None, :, :2], warp_mat
        #         )[0]
        #     else:
        #         transformed_proj_verts_2d = proj_verts_2d

        #     # Normalize to [-1, 1]
        #     normalized_proj_verts = transformed_proj_verts_2d.copy()
        #     normalized_proj_verts[:, 0] = 2.0 * normalized_proj_verts[:, 0] / img_w - 1.0
        #     normalized_proj_verts[:, 1] = 2.0 * normalized_proj_verts[:, 1] / img_h - 1.0
        #     if normalized_proj_verts.shape[1] == 2:
        #         proj_verts_vis = proj_verts[:, 2:3] if proj_verts.shape[1] > 2 else np.ones((proj_verts.shape[0], 1))
        #         normalized_proj_verts = np.hstack([normalized_proj_verts, proj_verts_vis])
        #     item['proj_verts'] = torch.from_numpy(normalized_proj_verts).float()

        item["person_valid"] = torch.ones((1, 1))

        # Store all transform outputs
        for key in data:
            item[key] = data[key]

        item["mask"] = item["mask"].float().unsqueeze(-3) * -1.0 # N, 1, H, W
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
        if self.is_train:
            if "cam_int" in self.data.files:
                item["focal_length"] = torch.tensor(
                    [self.cam_int[index][0, 0], self.cam_int[index][1, 1]]
                )
            if self.dataset == "3dpw-train-smplx":
                item["focal_length"] = torch.tensor([1961.1, 1969.2])
            # Will be 0 for 3dpw-train-smplx
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
        item["gender"] = self.gender[index]
        item["sample_index"] = index
        item["dataset_name"] = self.dataset
        if not self.is_train:
            if "3dpw" in self.dataset:
                if self.gender[index] == 1:
                    gt_smpl_out = self.smpl_female(
                        global_orient=item["pose"].unsqueeze(0)[:, :3],
                        body_pose=item["pose"].unsqueeze(0)[:, 3:],
                        betas=item["betas"].unsqueeze(0),
                    )
                    gt_vertices = gt_smpl_out.vertices
                else:
                    gt_smpl_out = self.smpl_male(
                        global_orient=item["pose"].unsqueeze(0)[:, :3],
                        body_pose=item["pose"].unsqueeze(0)[:, 3:],
                        betas=item["betas"].unsqueeze(0),
                    )
                    gt_vertices = gt_smpl_out.vertices

                item["vertices"] = gt_vertices[0].float()
            elif "rich" in self.dataset:
                if self.gender[index] == 1:
                    model = self.smpl_female
                    gt_smpl_out = self.smplx_female(
                        global_orient=item["pose"].unsqueeze(0)[:, :3],
                        body_pose=item["pose"].unsqueeze(0)[
                            :, 3 : NUM_JOINTS_SMPLX * 3
                        ],
                        betas=item["betas"].unsqueeze(0),
                    )
                    gt_vertices = gt_smpl_out.vertices
                else:
                    model = self.smpl_male
                    gt_smpl_out = self.smplx_male(
                        global_orient=item["pose"].unsqueeze(0)[:, :3],
                        body_pose=item["pose"].unsqueeze(0)[
                            :, 3 : NUM_JOINTS_SMPLX * 3
                        ],
                        betas=item["betas"].unsqueeze(0),
                    )
                    gt_vertices = gt_smpl_out.vertices

                gt_vertices = torch.matmul(self.smplx2smpl, gt_vertices)
                item["joints"] = torch.matmul(model.J_regressor, gt_vertices[0])
                item["vertices"] = gt_vertices[0].float()
            elif "h36m" in self.dataset:
                item["joints"] = self.joints[index]
                item["vertices"] = torch.zeros((6890, 3)).float()
            else:
                item["vertices"] = torch.zeros((6890, 3)).float()

        if not self.is_train:
            item["dataset_index"] = self.options.VAL_DS.split("_").index(self.dataset)

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
        self.mask_dir = self.img_dir.replace('training_images', 'masks').replace('_6fps/', '/').replace('png', 'masks')
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

        self.pose_cam = self.data["pose_cam"][:, : NUM_JOINTS_SMPLX * 3].astype(float)
        self.betas = self.data["shape"].astype(float)

        # Load keypoints
        full_joints = self.data["gtkps"]
        self.keypoints = full_joints[:, :24]

        self.mhr_keypoints_2d = self.data["mhr_keypoints_2d"]

        # Setup transforms (no augmentation for evaluation)
        input_size = (256, 256)
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
        maskname = os.path.join(self.mask_dir, self.imgname[index][:-4] + '_env.png')
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

        item["mask"] = item["mask"].float().unsqueeze(-3) * -1.0 # N, 1, H, W
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
