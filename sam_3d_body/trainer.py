import os
import cv2
import numpy as np
import torch
from typing import Dict, Optional
from collections import defaultdict
import roma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sam_3d_body.visualization.renderer import Renderer

from yacs.config import CfgNode
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader

from .models.meta_arch.sam3d_body import SAM3DBody
from .models.meta_arch.base_lightning_module import BaseLightningModule
from .losses.loss import Loss
from .data.bedlam_dataset import DatasetHMR as BEDLAMDataset
from .data.bedlam_dataset import MultiViewEvaluationDataset
from .metrics.metrics_tracker import Metrics
from .visualization.my_vis import Visualiser
from .configs.config import INDICES_PATH

import sys
from pathlib import Path

# Add project root to path for tools import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.vis_utils import my_visualize
from tools.vis_utils import my_visualize_samples
from tools.vis_utils import LIGHT_BLUE


class Trainer(BaseLightningModule):
    """
    Trainer class that extends SAM3DBody with PyTorch Lightning training logic.
    Inherits all model functionality from SAM3DBody.
    """

    def __init__(
        self, cfg: CfgNode, vis_save_dir: str = None, stack_vertically: bool = True
    ):
        super().__init__()

        self.cfg = cfg
        self.vis_save_dir = vis_save_dir
        self.stack_vertically = stack_vertically

        # Select model based on config
        self.model_type = cfg.TRAIN.get("MODEL_TYPE", "full")
        if self.model_type == "toy":
            assert False
            self.model = ToyModel(cfg)
        elif self.model_type == "full":
            self.model = SAM3DBody(cfg)
        else:
            raise ValueError(
                f"Unknown MODEL_TYPE: {self.model_type}. Must be 'toy' or 'full'."
            )

        self.metrics = Metrics()

        # self.val_ds = self.val_dataset()
        # self.train_ds = self.train_dataset()

        # Optionally enable dense keypoints based on config; if disabled, the model
        # will only use the canonical 70 MHR keypoints.
        self.use_dense_keypoints = bool(
            getattr(self.cfg.MODEL, "DENSE_KEYPOINTS", False)
        )
        self.mhr_dense_kp_indices = None
        if self.use_dense_keypoints:
            mhr_dense_kp_indices_np = np.load(INDICES_PATH)
            self.mhr_dense_kp_indices = torch.from_numpy(mhr_dense_kp_indices_np).long()
            # Expose to the meta-arch and the MHR head for dense keypoint extraction
            setattr(self.model, "mhr_dense_kp_indices", self.mhr_dense_kp_indices)
            setattr(
                self.model.head_pose, "mhr_dense_kp_indices", self.mhr_dense_kp_indices
            )

        # Load checkpoint only for full model (toy model doesn't have pretrained weights)
        if self.model_type == "full":
            checkpoint = torch.load(
                cfg.TRAIN.CKPT_PATH, map_location="cpu", weights_only=False
            )
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)

            if self.cfg.TRAIN.FREEZE_BACKBONE:
                for param in self.model.parameters():
                    param.requires_grad = False
                    
            for param in [
                self.model.head_pose.shape_uncertainty_proj,
                self.model.head_pose.scale_uncertainty_proj,
                self.model.head_pose.pose_3dof_uncertainty_proj,
                self.model.head_pose.pose_1dof_uncertainty_proj,
            ]:
                for p in param.parameters():
                    p.requires_grad = True
            if self.model.use_uncertainty_token:
                self.model.uncert_token.requires_grad = True

        self.scale_mean = self.model.head_pose.scale_mean.float()
        self.scale_comps = self.model.head_pose.scale_comps.float()

        self.criterion = Loss(
            cfg, scale_mean=self.scale_mean, scale_comps=self.scale_comps
        )

        self.faces = self.model.head_pose.faces.cpu().detach().numpy()

        self.visualiser = Visualiser(vis_save_dir, cfg=cfg, faces=self.faces)

    def training_step(self, batch: Dict, batch_idx: int):
        batch = self.preprocess(batch)

        outputs = self(batch, num_samples=5)

        loss_dict = self.criterion(outputs, batch)

        metrics = self.metrics(outputs, batch)

        self.log_and_visualise(
            loss_dict, metrics, batch, outputs, prefix="train_", batch_idx=batch_idx
        )

        # for k, v in loss_dict.items():
        #     print(f'{k}: {v.item():.3f}', end=' ')
        # print('')
        # import ipdb; ipdb.set_trace()

        return loss_dict["total_loss"]

    def log_and_visualise(
        self,
        loss_dict: Dict,
        metrics: Dict,
        batch: Dict,
        outputs: Dict,
        prefix: str = "",
        batch_idx: Optional[int] = None,
    ):

        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        loss_dict = {f"{prefix}{k}": v for k, v in loss_dict.items()}

        self.log(
            "pampjpe",
            metrics[f"{prefix}pampjpe"],
            prog_bar=(prefix == "train_"),
            logger=False,
        )
        self.log(
            "pampjpe_samples",
            metrics[f"{prefix}pampjpe_samples"],
            prog_bar=(prefix == "train_"),
            logger=False,
        )

        self.log(f"{prefix}loss", loss_dict[f"{prefix}total_loss"], prog_bar=True)
        self.log_dict(metrics, sync_dist=True)
        self.log_dict(loss_dict, sync_dist=True)


        if prefix == "train_":
            vis_step = int(self.global_step)
        else:
            vis_step = int(self.global_step if self.global_step > 0 else (batch_idx or 0))

        should_visualize = vis_step in [0, 250, 500, 1000, 2000, 3000, 4000] or (
            vis_step > 4000 and vis_step % 5000 == 0
        )
        global_rank = getattr(self, "global_rank", 0)
        if should_visualize and global_rank == 0:
            # if True:
            image = batch["img_ori"][0].data  # H W 3, bedlam 720 1280 3
            # image = batch['img'][0,0].data # [3, 256, 256] - CHW format, normalized
            image = image.cpu().detach().numpy()  # [3, H, W]

            image_crop = batch["img"][0,0].data.cpu().detach().numpy().transpose(1, 2, 0) * 255.0

            # Generate visualizations
            rend_img = my_visualize(
                image, outputs, self.faces, stack_vertically=self.stack_vertically
            )
            affine = batch["affine_trans"][0, 0]
            img_size = batch["img_size"][0, 0]
            rend_img_samples_crops = my_visualize_samples(
                image,
                outputs,
                self.faces,
                stack_vertically=self.stack_vertically,
                affine=affine,
                img_size=img_size,
            )
            rend_img_samples = my_visualize_samples(
                image,
                outputs,
                self.faces,
                stack_vertically=self.stack_vertically,
            )
            
            rend_img_bgr = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
            rend_img_samples_bgr = cv2.cvtColor(rend_img_samples, cv2.COLOR_RGB2BGR)
            rend_img_samples_crops_bgr = cv2.cvtColor(rend_img_samples_crops, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.vis_save_dir, f"{vis_step:06d}_img.png"),
                rend_img_bgr,
            )
            cv2.imwrite(
                os.path.join(self.vis_save_dir, f"{vis_step:06d}_samples.png"),
                rend_img_samples_bgr,
            )
            cv2.imwrite(
                os.path.join(self.vis_save_dir, f"{vis_step:06d}_samples_crops.png"),
                rend_img_samples_crops_bgr,
            )
            self.visualiser.visualise(
                outputs, batch, batch_idx=batch_idx, global_step=vis_step
            )
        return None

    def forward(self, batch: Dict, num_samples: int = 0) -> Dict:
        return self.model(batch, num_samples)

    def validation_step(self, batch: Dict, batch_idx: int):
        batch = self.preprocess(batch)

        outputs = self(batch, num_samples=5)

        loss_dict = self.criterion(outputs, batch)

        metrics = self.metrics(outputs, batch)

        self.log_and_visualise(
            loss_dict, metrics, batch, outputs, prefix="val_", batch_idx=batch_idx
        )

        return loss_dict["total_loss"]

    def test_step(self, batch: Dict, batch_idx: int):
        """
        Test step that collects metrics for each batch.
        Metrics are aggregated and printed at the end of the test epoch.
        """
        batch = self.preprocess(batch)

        outputs = self(batch, num_samples=5)

        loss_dict = self.criterion(outputs, batch)

        metrics = self.metrics(outputs, batch)

        self.log_and_visualise(
            loss_dict, metrics, batch, outputs, prefix="test_", batch_idx=batch_idx
        )

        return loss_dict["total_loss"]

    def preprocess(self, batch: Dict):
        mhr_model = self.model.head_pose

        gt_mhr_output = mhr_model.mhr(
            identity_coeffs=batch["shape_params"],
            model_parameters=batch["model_params"],
            face_expr_coeffs=batch["face_expr_coeffs"],
        )
        gt_verts, gt_skeleton_state = gt_mhr_output
        gt_joint_coords, gt_joint_quats, _ = torch.split(
            gt_skeleton_state, [3, 4, 1], dim=2
        )
        gt_verts = gt_verts / 100
        gt_joint_coords = gt_joint_coords / 100

        gt_vert_joints = torch.cat(
            [gt_verts, gt_joint_coords], dim=1
        )  # B x (num_verts + 127) x 3
        gt_keypoints_3d = (
            (mhr_model.keypoint_mapping @ gt_vert_joints.permute(1, 0, 2).flatten(1, 2))
            .reshape(-1, gt_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )
        if batch["dataset_name"][0] == "4d-dress":
            R = batch["cam_ext"][:, :3, :3]
            gt_verts = gt_verts @ R.transpose(-2, -1)

        batch["gt_verts_w_transl"] = gt_verts

        cam_int = batch["cam_int"]
        if "cam_ext" not in batch:
            # SSP-3D
            assert batch["dataset_name"][0] == "ssp3d"
            trans_cam = batch["trans_cam"]
        else:
            cam_ext = batch["cam_ext"]
            trans_cam = cam_ext[:, :3, 3]

        def project(points, cam_trans, cam_int):
            points = points + cam_trans
            # Normalize by Z (divide by last coordinate)
            projected_points = points / points[..., -1].unsqueeze(-1)
            # Multiply by camera intrinsics: cam_int @ projected_points.T
            projected_points = torch.einsum("bij, bkj->bki", cam_int, projected_points)
            return projected_points

        keypoints_2d_by_projection = project(
            gt_keypoints_3d, trans_cam.unsqueeze(1), cam_int
        )[:, :70, :2]

        # verts_2d = project(
        #     gt_verts, trans_cam.unsqueeze(1), cam_int
        # )[:, :, :2]

        # # import matplotlib.pyplot as plt
        # # plt.imshow(batch["img_ori"][0].data.cpu().numpy())
        # # plt.scatter(verts_2d[0,:, 0].cpu().numpy(), verts_2d[0,:, 1].cpu().numpy(), s=0.1, c='red')
        # # plt.title("2D projected mesh vertices")
        # # plt.axis("off")
        # # plt.savefig("temp_vis.png")
        # # plt.close()

        # # import ipdb; ipdb.set_trace()

        # Ground-truth 2D keypoints: 70 canonical + optional dense vertices projected
        if self.use_dense_keypoints and self.mhr_dense_kp_indices is not None:
            dense_kp2d = project(
                gt_verts[:, self.mhr_dense_kp_indices, :],
                trans_cam.unsqueeze(1),
                cam_int,
            )[:, :, :2]
            kp2d = torch.cat([keypoints_2d_by_projection, dense_kp2d], dim=1)
        else:
            kp2d = keypoints_2d_by_projection

        gt_kp2d_h = torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=-1).float()
        affine = batch["affine_trans"][:, 0].float()
        img_size = batch["img_size"][:, 0]

        gt_kp2d_crop = gt_kp2d_h @ affine.mT  # [B, 70, 3] @ [B, 3, 2] = [B, 70, 2]
        gt_kp2d_crop = gt_kp2d_crop[..., :2]

        # Normalize to [-0.5, 0.5], same as _full_to_crop
        # img_size needs to be [B, 1, 2] for proper broadcasting
        gt_kp2d_crop = gt_kp2d_crop / img_size.unsqueeze(1) - 0.5  # [B, 70, 2]

        batch["keypoints_2d"] = gt_kp2d_crop

        model_parameters = batch["model_params"]
        model_parameters[:, :3] = 0
        global_rot = batch["model_params"][:, 3:6]

        # Add 180-degree rotation around X-axis
        global_rot_mat = roma.euler_to_rotmat("xyz", global_rot)  # B x 3 x 3

        batch_size = global_rot.shape[0]
        rot_180_x = (
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=global_rot.dtype,
                device=global_rot.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Multiply rotations: new_rot = rot_180_x @ global_rot_mat
        new_global_rot_mat = torch.bmm(rot_180_x, global_rot_mat)

        # Convert back to Euler angles
        global_rot = roma.rotmat_to_euler("xyz", new_global_rot_mat)

        # Update model_parameters with the new global_rot
        model_parameters[:, 3:6] = global_rot

        gt_mhr_output = mhr_model.mhr(
            identity_coeffs=batch["shape_params"],
            model_parameters=model_parameters,
            face_expr_coeffs=batch["face_expr_coeffs"],
        )
        gt_verts, gt_skeleton_state = gt_mhr_output
        gt_joint_coords, gt_joint_quats, _ = torch.split(
            gt_skeleton_state, [3, 4, 1], dim=2
        )
        gt_verts = gt_verts / 100
        gt_joint_coords = gt_joint_coords / 100

        gt_vert_joints = torch.cat(
            [gt_verts, gt_joint_coords], dim=1
        )  # B x (num_verts + 127) x 3
        gt_keypoints_3d_all = (
            (mhr_model.keypoint_mapping @ gt_vert_joints.permute(1, 0, 2).flatten(1, 2))
            .reshape(-1, gt_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )

        # Ground-truth 3D keypoints: always include the canonical 70 MHR keypoints,
        # and optionally append dense keypoints if enabled.
        gt_kp3d_70 = gt_keypoints_3d_all[:, :70]  # [B, 70, 3]
        if self.use_dense_keypoints and self.mhr_dense_kp_indices is not None:
            dense_kp3d_gt = gt_verts[:, self.mhr_dense_kp_indices, :]  # [B, N_dense, 3]
            gt_keypoints_3d = torch.cat(
                [gt_kp3d_70, dense_kp3d_gt], dim=1
            )  # [B, 70+N_dense, 3]
        else:
            gt_keypoints_3d = gt_kp3d_70

        batch["joints_3d"] = gt_joint_coords
        batch["vertices"] = gt_verts
        batch["keypoints_3d"] = gt_keypoints_3d

        return batch

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.TRAIN.LR)
        return optimizer


    def train_dataset(self):
        options = self.cfg.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split("_")
        dataset_list = [BEDLAMDataset(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)

        return train_ds

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.cfg.DATASET.BATCH_SIZE,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
            pin_memory=self.cfg.DATASET.PIN_MEMORY,
            shuffle=False,  # self.cfg.DATASET.SHUFFLE_TRAIN,
            drop_last=True,
        )

    def val_dataset(self):
        datasets = self.cfg.DATASET.VAL_DS.split("_")
        # logger.info(f"Validation datasets are: {datasets}")
        # val_datasets = []
        # for dataset_name in datasets:
        #     val_datasets.append(
        #         BEDLAMDataset(
        #             options=self.cfg.DATASET,
        #             dataset=dataset_name,
        #             is_train=False,
        #         )
        #     )
        val_datasets = [BEDLAMDataset(self.cfg.DATASET, ds) for ds in datasets]
        val_ds = ConcatDataset(val_datasets)
        return val_ds

    def val_dataloader(self):
        self.val_ds = self.val_dataset()
        # dataloaders = []
        # for val_ds in self.val_ds:
        #     dataloaders.append(
        #         DataLoader(
        #             dataset=val_ds,
        #             batch_size=self.cfg.DATASET.BATCH_SIZE,
        #             shuffle=False,
        #             num_workers=self.cfg.DATASET.NUM_WORKERS,
        #             drop_last=True,
        #         )
        #     )
        # return dataloaders
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.cfg.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
            pin_memory=self.cfg.DATASET.PIN_MEMORY,
            drop_last=True,
        )

    # def test_dataset(self):
    #     """
    #     Create test dataset. Uses TEST_DS from config if available, otherwise falls back to VAL_DS.
    #     """
    #     # Check if TEST_DS is configured, otherwise use VAL_DS
    #     if hasattr(self.cfg.DATASET, "TEST_DS") and self.cfg.DATASET.TEST_DS:
    #         datasets = self.cfg.DATASET.TEST_DS.split("_")
    #         logger.info(f"Test datasets are: {datasets}")
    #     else:
    #         datasets = self.cfg.DATASET.VAL_DS.split("_")
    #         logger.info(f"Test datasets (using VAL_DS): {datasets}")

    #     test_datasets = []
    #     for dataset_name in datasets:
    #         test_datasets.append(
    #             BEDLAMDataset(
    #                 options=self.cfg.DATASET,
    #                 dataset=dataset_name,
    #                 is_train=False,
    #             )
    #         )
    #     return test_datasets

    # def test_dataloader(self):
    #     """
    #     Create test dataloader. Returns a list of dataloaders, one for each test dataset.
    #     """
    #     if not hasattr(self, "test_ds"):
    #         self.test_ds = self.test_dataset()

    #     dataloaders = []
    #     for test_ds in self.test_ds:
    #         dataloaders.append(
    #             DataLoader(
    #                 dataset=test_ds,
    #                 batch_size=self.cfg.DATASET.BATCH_SIZE,
    #                 shuffle=False,
    #                 num_workers=self.cfg.DATASET.NUM_WORKERS,
    #                 drop_last=False,  # Don't drop last batch in test to evaluate all samples
    #             )
    #         )
    #     return dataloaders

    def multiview_eval_dataset(self, num_view: int = 4, dataset_name: str = "4d-dress"):
        """
        Build a BEDLAM multi-view evaluation dataset using MultiViewEvaluationDataset.

        Each sample corresponds to a unique serial number (serno) and contains
        `num_view` different camera views of the same subject.
        """
        if dataset_name is not None:
            self.cfg.DATASET.VAL_DS = dataset_name

        if self.cfg.DATASET.VAL_DS == "ssp3d":
            from sam_3d_body.data.ssp3d_dataset import MultiSSP3DDataset
            logger.info(f"SSP-3D dataset with num_view={num_view}")
            return MultiSSP3DDataset(
                "/scratches/kyuban/cq244/datasets/SSP-3D/ssp_3d", num_view=num_view, cfg=self.cfg
            )
        elif self.cfg.DATASET.VAL_DS == "4d-dress":
            from sam_3d_body.data.d4dress_dataset import MultiD4DressDataset

            logger.info(f"4D-DRESS dataset with num_view={num_view}")
            ids = [
                "00122","00123","00127","00129","00134",
                "00135","00136","00137","00140","00147",
                "00148","00149","00151","00152","00154",
                "00156","00160","00163","00167","00168",
                "00169","00170","00174","00175","00176",
                "00179","00180","00185","00187","00190",
            ]
            return MultiD4DressDataset(ids, cfg=self.cfg)

        dataset_names = self.cfg.DATASET.VAL_DS.split("_")
        dataset_name = dataset_names[0]

        logger.info(
            f"Creating MultiViewEvaluationDataset for '{dataset_name}' "
            f"with num_view={num_view}"
        )

        multiview_ds = MultiViewEvaluationDataset(
            options=self.cfg.DATASET,
            dataset=dataset_name,
            num_view=num_view,
            is_train=True,  # uses training BEDLAM splits
        )

        return multiview_ds

    def multiview_eval_dataloader(
        self, num_view: int = 4, batch_size: int = 1, dataset_name: str = "4d-dress"
    ):
        """
        DataLoader wrapping the multi-view evaluation dataset.

        Batch size defaults to 1 so that each batch corresponds to a single serno,
        with `num_view` views.
        """
        multiview_ds = self.multiview_eval_dataset(
            num_view=num_view, dataset_name=dataset_name
        )
        loader = DataLoader(
            dataset=multiview_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
            pin_memory=self.cfg.DATASET.PIN_MEMORY,
            drop_last=False,
        )
        return loader

    def run_multiview_prediction(
        self,
        num_view: int = 4,
        max_batches: Optional[int] = None,
        dataset_name: str = "4d-dress",
    ):
        """
        Run MHR predictions for each view loaded by MultiViewEvaluationDataset.

        For each serno, iterates over all views and runs the MHR model, keeping
        track of all predicted values.

        Returns:
            A list of dictionaries, one per serno in the dataset:
                {
                    "serno": <serno_id>,
                    "indices": [idx_0, idx_1, ...],   # original BEDLAM indices
                    "pred_vertices": [V x (N_v, 3)],
                    "pred_joints":   [V x (N_j, 3)],
                }
        """
        # Get device from model parameters (works even when called outside Lightning training loop)
        device = self.device

        dataloader = self.multiview_eval_dataloader(
            num_view=num_view, batch_size=1, dataset_name=dataset_name
        )

        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move tensor fields to device
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Batch size is 1 by construction
            # Shapes: [1, V, ...] -> we work over the view dimension
            num_views = batch["num_views"][0].item()
            # serno = batch["selected_serno"][0].item()
            # indices = batch["selected_indices"][0].tolist()

            # if the value is a tensor and its first two dims are [batch, num_views], flatten
            bs, num_views = batch["img"].shape[:2]
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_views:
                        batch[k] = v.flatten(0, 1)

            batch = self.preprocess(batch)

            with torch.no_grad():
                outputs = self.model(batch, num_samples=0)

            pred_shape = outputs["mhr"]["shape"]
            pred_scale = outputs["mhr"]["scale"]
            shape_var = outputs["mhr"]["shape_uncertainty"]
            scale_var = outputs["mhr"]["scale_uncertainty"]

            # shape_var: [batch, D], want [batch, D, D] with diag elements
            shape_var_diag = torch.diag_embed(shape_var)
            scale_var_diag = torch.diag_embed(scale_var)

            pred_shape = pred_shape.unflatten(0, (bs, num_views))
            pred_scale = pred_scale.unflatten(0, (bs, num_views))
            shape_var_diag = shape_var_diag.unflatten(0, (bs, num_views))
            scale_var_diag = scale_var_diag.unflatten(0, (bs, num_views))

            shape_mu_star, shape_sigma_star = self.merge_predictions_batch(
                pred_shape, shape_var_diag
            )

            # Variance for per-view and merged shape parameters
            shape_var_unflattened = shape_var.unflatten(0, (bs, num_views))
            merged_shape_var = torch.diagonal(shape_sigma_star, dim1=-2, dim2=-1)

            indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
            scale_mu_star, scale_sigma_star = self.merge_predictions_batch(
                pred_scale[..., indices], scale_var_diag
            )
            scale_mu_star_full = pred_scale.mean(dim=1)
            scale_mu_star_full[..., indices] = scale_mu_star

            shape_mean = pred_shape.mean(dim=1).repeat_interleave(
                num_views, dim=0
            )  # naive average of parameters
            scale_mean = pred_scale.mean(dim=1).repeat_interleave(num_views, dim=0)

            mean_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mean,
                scale_params=scale_mean,
                global_trans=torch.zeros_like(outputs["mhr"]["global_rot"]),
                global_rot=outputs["mhr"]["global_rot"],
                body_pose_params=outputs["mhr"]["body_pose"],
                hand_pose_params=outputs["mhr"]["hand"],
                expr_params=outputs["mhr"]["face"],
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
                do_pcblend=True,
            )
            verts_mean, j3d_mean, jcoords_mean, mhr_model_params, joint_global_rots = (
                mean_mhr_output
            )
            verts_mean[..., [1, 2]] *= -1
            j3d_mean[..., [1, 2]] *= -1

            merged_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mu_star.repeat_interleave(num_views, dim=0),
                scale_params=scale_mu_star_full.repeat_interleave(num_views, dim=0),
                global_trans=torch.zeros_like(outputs["mhr"]["global_rot"]),
                global_rot=outputs["mhr"]["global_rot"],
                body_pose_params=outputs["mhr"]["body_pose"],
                hand_pose_params=outputs["mhr"]["hand"],
                expr_params=outputs["mhr"]["face"],
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
                do_pcblend=True,
            )
            verts_star, j3d_star, jcoords_star, mhr_model_params, joint_global_rots = (
                merged_mhr_output
            )
            verts_star[..., [1, 2]] *= -1
            j3d_star[..., [1, 2]] *= -1

            # get neutral gt params
            gt_shape = batch["shape_params"]
            gt_model_params = batch["model_params"]
            gt_face_params = batch["face_expr_coeffs"]
            gt_model_params[:, :-68] = torch.zeros_like(gt_model_params[:, :-68])
            gt_face_params = torch.zeros_like(gt_face_params)
            gt_neutral_mhr_output = self.model.head_pose.mhr(
                gt_shape, gt_model_params, gt_face_params
            )
            gt_neutral_verts, gt_neutral_skeleton_state = gt_neutral_mhr_output
            gt_neutral_joint_coords, _, _ = torch.split(
                gt_neutral_skeleton_state, [3, 4, 1], dim=2
            )
            gt_neutral_verts = gt_neutral_verts / 100
            gt_neutral_joint_coords = gt_neutral_joint_coords / 100

            # get neutral pred
            per_view_neutral_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=pred_shape.flatten(0, 1),
                scale_params=pred_scale.flatten(0, 1),
                global_trans=torch.zeros_like(outputs["mhr"]["global_rot"]),
                global_rot=torch.zeros_like(outputs["mhr"]["global_rot"]),
                body_pose_params=torch.zeros_like(outputs["mhr"]["body_pose"]),
                hand_pose_params=torch.zeros_like(outputs["mhr"]["hand"]),
                expr_params=torch.zeros_like(outputs["mhr"]["face"]),
                return_joint_coords=True,
            )
            per_view_neutral_verts, per_view_neutral_joint_coords = (
                per_view_neutral_mhr_output
            )

            # get merged neutral pred
            merged_neutral_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mu_star.repeat_interleave(num_views, dim=0),
                scale_params=scale_mu_star_full.repeat_interleave(num_views, dim=0),
                global_trans=torch.zeros_like(outputs["mhr"]["global_rot"]),
                global_rot=torch.zeros_like(outputs["mhr"]["global_rot"]),
                body_pose_params=torch.zeros_like(outputs["mhr"]["body_pose"]),
                hand_pose_params=torch.zeros_like(outputs["mhr"]["hand"]),
                expr_params=torch.zeros_like(outputs["mhr"]["face"]),
                return_joint_coords=True,
            )
            merged_neutral_verts, merged_neutral_joint_coords = (
                merged_neutral_mhr_output
            )

            mean_neutral_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mean,
                scale_params=scale_mean,
                global_trans=torch.zeros_like(outputs["mhr"]["global_rot"]),
                global_rot=torch.zeros_like(outputs["mhr"]["global_rot"]),
                body_pose_params=torch.zeros_like(outputs["mhr"]["body_pose"]),
                hand_pose_params=torch.zeros_like(outputs["mhr"]["hand"]),
                expr_params=torch.zeros_like(outputs["mhr"]["face"]),
                return_joint_coords=True,
            )
            mean_neutral_verts, mean_neutral_joint_coords = mean_neutral_mhr_output

            # ----- mpjpe -----
            # per-view
            per_view_mpjpe = torch.sqrt(
                ((per_view_neutral_joint_coords - gt_neutral_joint_coords) ** 2).sum(
                    dim=-1
                )
            ).mean(dim=1)
            # merged
            merged_mpjpe = torch.sqrt(
                ((merged_neutral_joint_coords - gt_neutral_joint_coords) ** 2).sum(
                    dim=-1
                )
            ).mean(dim=1)
            # mean
            mean_mpjpe = torch.sqrt(
                ((mean_neutral_joint_coords - gt_neutral_joint_coords) ** 2).sum(dim=-1)
            ).mean(dim=1)

            # ----- pve -----
            # per-view
            per_view_pve = torch.sqrt(
                ((per_view_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)
            ).mean(dim=1)
            # merged
            merged_pve = torch.sqrt(
                ((merged_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)
            ).mean(dim=1)
            # mean
            mean_pve = torch.sqrt(
                ((mean_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)
            ).mean(dim=1)

            # ----- pampjpe -----
            from sam_3d_body.metrics.metrics_tracker import reconstruction_error
            from sam_3d_body.metrics.metrics_tracker import (
                scale_and_translation_transform_batch,
            )

            # per-view
            per_view_pampjpe, _ = reconstruction_error(
                per_view_neutral_joint_coords.cpu().detach().numpy(),
                gt_neutral_joint_coords.cpu().detach().numpy(),
                reduction="none",
            )
            per_view_pampjpe = per_view_pampjpe.mean(axis=-1)
            # merged
            merged_pampjpe, _ = reconstruction_error(
                merged_neutral_joint_coords.cpu().detach().numpy(),
                gt_neutral_joint_coords.cpu().detach().numpy(),
                reduction="none",
            )
            merged_pampjpe = merged_pampjpe.mean(axis=-1)
            # mean
            mean_pampjpe, _ = reconstruction_error(
                mean_neutral_joint_coords.cpu().detach().numpy(),
                gt_neutral_joint_coords.cpu().detach().numpy(),
                reduction="none",
            )
            mean_pampjpe = mean_pampjpe.mean(axis=-1)

            # ----- pvetsc -----
            pred_sc = scale_and_translation_transform_batch(
                per_view_neutral_verts.cpu().detach().numpy(),
                gt_neutral_verts.cpu().detach().numpy(),
            )
            merged_sc = scale_and_translation_transform_batch(
                merged_neutral_verts.cpu().detach().numpy(),
                gt_neutral_verts.cpu().detach().numpy(),
            )
            mean_sc = scale_and_translation_transform_batch(
                mean_neutral_verts.cpu().detach().numpy(),
                gt_neutral_verts.cpu().detach().numpy(),
            )
            per_view_pvetsc = np.linalg.norm(
                pred_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
            ).mean(axis=1)
            merged_pvetsc = np.linalg.norm(
                merged_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
            ).mean(axis=1)
            mean_pvetsc = np.linalg.norm(
                mean_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
            ).mean(axis=1)

            # print(per_view_mpjpe.shape, mean_mpjpe.shape, merged_mpjpe.shape)
            # print(per_view_pve.shape, mean_pve.shape, merged_pve.shape)
            # print(per_view_pampjpe.shape, mean_pampjpe.shape, merged_pampjpe.shape)
            # print(per_view_pvetsc.shape, mean_pvetsc.shape, merged_pvetsc.shape)

            print(
                f"mpjpe: view avg: {per_view_mpjpe.mean():.4f}, view min: {per_view_mpjpe.min():.4f}, mean: {mean_mpjpe.mean():.4f} merged: {merged_mpjpe.mean():.4f}"
            )
            print(
                f"pve: view avg: {per_view_pve.mean():.4f}, view min: {per_view_pve.min():.4f}, mean: {mean_pve.mean():.4f}, merged: {merged_pve.mean():.4f}"
            )
            print(
                f"pampjpe: view avg: {per_view_pampjpe.mean():.4f}, view min: {per_view_pampjpe.min():.4f}, mean: {mean_pampjpe.mean():.4f}, merged: {merged_pampjpe.mean():.4f}"
            )
            print(
                f"pvetsc: view avg: {per_view_pvetsc.mean():.4f}, view min: {per_view_pvetsc.min():.4f}, mean: {mean_pvetsc.mean():.4f}, merged: {merged_pvetsc.mean():.4f}"
            )

            metrics["per_view_mpjpe"].append(per_view_mpjpe.mean().item())
            metrics["best_per_view_mpjpe"].append(per_view_mpjpe.min().item())
            metrics["mean_mpjpe"].append(mean_mpjpe.mean().item())
            metrics["merged_mpjpe"].append(merged_mpjpe.mean().item())

            metrics["per_view_pve"].append(per_view_pve.mean().item())
            metrics["best_per_view_pve"].append(per_view_pve.min().item())
            metrics["mean_pve"].append(mean_pve.mean().item())
            metrics["merged_pve"].append(merged_pve.mean().item())

            metrics["per_view_pampjpe"].append(per_view_pampjpe.mean().item())
            metrics["best_per_view_pampjpe"].append(per_view_pampjpe.min().item())
            metrics["mean_pampjpe"].append(mean_pampjpe.mean().item())
            metrics["merged_pampjpe"].append(merged_pampjpe.mean().item())

            metrics["per_view_pvetsc"].append(per_view_pvetsc.mean().item())
            metrics["best_per_view_pvetsc"].append(per_view_pvetsc.min().item())
            metrics["mean_pvetsc"].append(mean_pvetsc.mean().item())
            metrics["merged_pvetsc"].append(merged_pvetsc.mean().item())

            # Initialize gallery: list of lists to store rendered images [bs][num_views]
            gallery = [[None for _ in range(num_views)] for _ in range(bs)]

            renderer = Renderer(
                focal_length=outputs["mhr"]["focal_length"][0], faces=self.faces
            )

            for i in range(1):
                all_distances = []
                pred_vertex_dists = {}
                merged_vertex_dists = {}

                for view in range(num_views):
                    flat_idx = i * num_views + view

                    verts = (
                        outputs["mhr"]["pred_vertices"][flat_idx].cpu().detach().numpy()
                    )
                    gt_verts = (
                        batch["gt_verts_w_transl"][flat_idx].cpu().detach().numpy()
                    )
                    merged_verts = verts_star[flat_idx].cpu().detach().numpy()

                    pred_dist = np.linalg.norm(verts - gt_verts, axis=1)
                    merged_dist = np.linalg.norm(merged_verts - gt_verts, axis=1)

                    pred_vertex_dists[view] = pred_dist
                    merged_vertex_dists[view] = merged_dist

                    all_distances.append(pred_dist)
                    all_distances.append(merged_dist)

                all_distances = np.concatenate(all_distances)
                min_dist = float(all_distances.min()) if all_distances.size > 0 else 0.0
                max_dist = float(all_distances.max()) if all_distances.size > 0 else 1.0

                if max_dist > min_dist:
                    denom = max_dist - min_dist
                else:
                    denom = 1.0

                def build_vertex_colors(dists: np.ndarray) -> np.ndarray:
                    """Map distances to RGBA vertex colors using shared viridis scale."""
                    normalized = (dists - min_dist) / denom
                    normalized = np.clip(normalized, 0.0, 1.0)
                    colors_rgb = plt.cm.viridis(normalized)[..., :3]  # (V, 3)
                    vertex_colors = np.ones((colors_rgb.shape[0], 4), dtype=np.float32)
                    vertex_colors[:, :3] = colors_rgb
                    return vertex_colors

                # Second pass: render GT (solid color), and pred/merged with per-vertex viridis colors
                for view in range(num_views):
                    img_for_render = batch["img_ori"][view][i].cpu().detach().numpy()

                    flat_idx = i * num_views + view

                    verts = (
                        outputs["mhr"]["pred_vertices"][flat_idx].cpu().detach().numpy()
                    )
                    cam_t = (
                        outputs["mhr"]["pred_cam_t"][flat_idx].cpu().detach().numpy()
                    )

                    gt_verts = (
                        batch["gt_verts_w_transl"][flat_idx].cpu().detach().numpy()
                    )
                    # gt_verts[..., [1, 2]] *= -1
                    if "cam_ext" not in batch:
                        # SSP-3D
                        assert batch["dataset_name"][0] == "ssp3d"
                        gt_cam_t = batch["trans_cam"][flat_idx].cpu().detach().numpy()
                    else:
                        gt_cam_t = (
                            batch["cam_ext"][flat_idx][:3, -1].cpu().detach().numpy()
                        )

                    merged_verts = verts_star[flat_idx].cpu().detach().numpy()

                    # GT: keep fixed LIGHT_BLUE color
                    gt_rendered_img = (
                        renderer(
                            gt_verts,
                            gt_cam_t,
                            img_for_render.copy(),
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            camera_center=(
                                batch["cam_int"][flat_idx][0, 2],
                                batch["cam_int"][flat_idx][1, 2],
                            ),
                        )
                        * 255
                    ).astype(np.uint8)

                    # Predicted mesh: per-vertex viridis colors from distance to GT
                    pred_colors = build_vertex_colors(pred_vertex_dists[view])
                    rendered_img = (
                        renderer(
                            verts,
                            cam_t,
                            img_for_render.copy(),
                            mesh_base_color=(1.0, 0.8, 0.5),  # unused when vertex_colors is set
                            scene_bg_color=(1, 1, 1),
                            camera_center=(
                                batch["cam_int"][flat_idx][0, 2],
                                batch["cam_int"][flat_idx][1, 2],
                            ),
                            vertex_colors=pred_colors,
                        )
                        * 255
                    ).astype(np.uint8)

                    # Merged mesh: per-vertex viridis colors from distance to GT
                    merged_colors = build_vertex_colors(merged_vertex_dists[view])
                    rendered_merged_img = (
                        renderer(
                            merged_verts,
                            cam_t,
                            img_for_render.copy(),
                            mesh_base_color=(0.5, 1.0, 0.5),  # unused when vertex_colors is set
                            scene_bg_color=(1, 1, 1),
                            camera_center=(
                                batch["cam_int"][flat_idx][0, 2],
                                batch["cam_int"][flat_idx][1, 2],
                            ),
                            vertex_colors=merged_colors,
                        )
                        * 255
                    ).astype(np.uint8)

                    # Add text labels to images
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_scale_small = 0.6
                    thickness = 2
                    color = (255, 255, 255)  # White text
                    bg_color = (0, 0, 0)  # Black background for text

                    # Label gt image
                    gt_label = f"GT {view}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        gt_label, font, font_scale, thickness
                    )
                    cv2.rectangle(
                        gt_rendered_img,
                        (10, 10),
                        (10 + text_width + 4, 10 + text_height + baseline + 4),
                        bg_color,
                        -1,
                    )
                    cv2.putText(
                        gt_rendered_img,
                        gt_label,
                        (12, 10 + text_height),
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # Label GT image
                    gt_label = f"GT view {view}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        gt_label, font, font_scale, thickness
                    )
                    pampjpe_line = f"PA-MPJPE | View {view}: {per_view_pampjpe[view].item():.4f} | Merged: {merged_pampjpe.mean().item():.4f}"
                    pvetsc_line = f"PVE-T-SC | View {view}: {per_view_pvetsc[view].item():.4f} | Merged: {merged_pvetsc.mean().item():.4f}"
                    text_lines = [pampjpe_line, pvetsc_line]

                    y_start = 10 + text_height + baseline + 10
                    y_offset = y_start
                    for line in text_lines:
                        (tw, th), bl = cv2.getTextSize(
                            line, font, font_scale_small, thickness
                        )
                        cv2.rectangle(
                            gt_rendered_img,
                            (10, y_offset),
                            (10 + tw + 4, y_offset + th + bl + 4),
                            bg_color,
                            -1,
                        )
                        cv2.putText(
                            gt_rendered_img,
                            line,
                            (12, y_offset + th),
                            font,
                            font_scale_small,
                            color,
                            thickness,
                        )
                        y_offset += th + bl + 6

                    # Label predicted image
                    pred_label = f"Pred view {view}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        pred_label, font, font_scale, thickness
                    )
                    cv2.rectangle(
                        rendered_img,
                        (10, 10),
                        (10 + text_width + 4, 10 + text_height + baseline + 4),
                        bg_color,
                        -1,
                    )
                    cv2.putText(
                        rendered_img,
                        pred_label,
                        (12, 10 + text_height),
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # Add per-view predicted shape parameters (first 5) and uncertainties
                    pred_mu = pred_shape[i, view].cpu().detach().numpy()
                    pred_var = shape_var_unflattened[i, view].cpu().detach().numpy()
                    mu_str = "pred shape mean: " + " ".join(
                        f"{v:.2f}" for v in pred_mu[:5]
                    )
                    sigma_str = "pred shape var: " + " ".join(
                        f"{v:.2f}" for v in pred_var[:5]
                    )

                    pred_text_lines = [mu_str, sigma_str]
                    y_start = 10 + text_height + baseline + 10
                    y_offset = y_start
                    for line in pred_text_lines:
                        (tw, th), bl = cv2.getTextSize(
                            line, font, font_scale_small, thickness
                        )
                        cv2.rectangle(
                            rendered_img,
                            (10, y_offset),
                            (10 + tw + 4, y_offset + th + bl + 4),
                            bg_color,
                            -1,
                        )
                        cv2.putText(
                            rendered_img,
                            line,
                            (12, y_offset + th),
                            font,
                            font_scale_small,
                            color,
                            thickness,
                        )
                        y_offset += th + bl + 6

                    # Label merged image
                    merged_label = f"Merged view {view}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        merged_label, font, font_scale, thickness
                    )
                    cv2.rectangle(
                        rendered_merged_img,
                        (10, 10),
                        (10 + text_width + 4, 10 + text_height + baseline + 4),
                        bg_color,
                        -1,
                    )
                    cv2.putText(
                        rendered_merged_img,
                        merged_label,
                        (12, 10 + text_height),
                        font,
                        font_scale,
                        color,
                        thickness,
                    )

                    # Add merged shape parameters (first 5) and uncertainties
                    merged_mu = shape_mu_star[i].cpu().detach().numpy()
                    merged_var = merged_shape_var[i].cpu().detach().numpy()
                    m_mu_str = "merged shape mean: " + " ".join(
                        f"{v:.2f}" for v in merged_mu[:5]
                    )
                    m_sigma_str = "merged shape var: " + " ".join(
                        f"{v:.2f}" for v in merged_var[:5]
                    )

                    merged_text_lines = [m_mu_str, m_sigma_str]
                    y_start_m = 10 + text_height + baseline + 10
                    y_offset_m = y_start_m
                    for line in merged_text_lines:
                        (tw, th), bl = cv2.getTextSize(
                            line, font, font_scale_small, thickness
                        )
                        cv2.rectangle(
                            rendered_merged_img,
                            (10, y_offset_m),
                            (10 + tw + 4, y_offset_m + th + bl + 4),
                            bg_color,
                            -1,
                        )
                        cv2.putText(
                            rendered_merged_img,
                            line,
                            (12, y_offset_m + th),
                            font,
                            font_scale_small,
                            color,
                            thickness,
                        )
                        y_offset_m += th + bl + 6

                    gallery[i][view] = np.concatenate(
                        [gt_rendered_img, rendered_img, rendered_merged_img], axis=1
                    )

            gallery_rows = []
            for i in range(1):
                row = np.concatenate(
                    [gallery[i][view] for view in range(num_views)], axis=0
                )
                gallery_rows.append(row)

            # gallery_img = np.concatenate(gallery_rows, axis=0)
            gallery_img = gallery_rows[0]
            gallery_img_bgr = cv2.cvtColor(gallery_img, cv2.COLOR_RGB2BGR)

            save_dir = self.vis_save_dir if self.vis_save_dir else "."
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"multiview_batch{batch_idx:03d}_bs{bs}_views{num_views}.png",
            )
            cv2.imwrite(save_path, gallery_img_bgr)
            logger.info(
                f"Saved multiview gallery: {save_path} (shape: {gallery_img.shape})"
            )

            neutral_renderer = Renderer(focal_length=512, faces=self.faces)
            generic_cam_t = np.array([0.0, 0.75, 2.5])

            # Render GT neutral mesh
            gt_neutral_verts = gt_neutral_verts.cpu().detach().numpy()
            gt_neutral_verts[..., [1, 2]] *= -1
            gt_neutral_rendered = (
                neutral_renderer(
                    gt_neutral_verts[view],
                    generic_cam_t,
                    np.ones((512, 512, 3)) * 255,
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                * 255
            ).astype(np.uint8)

            text_config = {
                "org": (12, 10 + text_height),
                "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale": 1.0,
                "color": (0, 0, 0),
                "thickness": 2,
            }
            cv2.putText(
                gt_neutral_rendered,
                "GT",
                **text_config,
            )

            # Render per-view neutral meshes (4 views)
            per_view_rendered = []
            colors = [
                (1.0, 0.8, 0.5),  # light orange
                (0.8, 0.5, 1.0),  # light purple
                (0.5, 0.8, 1.0),  # light blue
                (1.0, 0.5, 0.5),  # light red
            ]
            per_view_verts = per_view_neutral_verts.cpu().detach().numpy()
            per_view_verts[..., [1, 2]] *= -1
            for view in range(num_views):
                rendered = (
                    neutral_renderer(
                        per_view_verts[view],
                        generic_cam_t,
                        np.ones((512, 512, 3)) * 255,
                        mesh_base_color=colors[view % len(colors)],
                        scene_bg_color=(1, 1, 1),
                    )
                    * 255
                ).astype(np.uint8)
                per_view_rendered.append(rendered)
                cv2.putText(
                    rendered,
                    f"View {view}",
                    **text_config,
                )
            # Render merged neutral mesh
            merged_verts = merged_neutral_verts[0].cpu().detach().numpy()
            merged_verts[..., [1, 2]] *= -1
            merged_neutral_rendered = (
                neutral_renderer(
                    merged_verts,
                    generic_cam_t,
                    np.ones((512, 512, 3)) * 255,
                    mesh_base_color=(0.5, 1.0, 0.5),  # light green
                    scene_bg_color=(1, 1, 1),
                )
                * 255
            ).astype(np.uint8)

            cv2.putText(
                merged_neutral_rendered,
                "Merged",
                **text_config,
            )

            gallery_images = (
                [gt_neutral_rendered] + [merged_neutral_rendered] + per_view_rendered
            )

            # Concatenate horizontally
            gallery_img = np.concatenate(gallery_images, axis=1)
            gallery_img_bgr = cv2.cvtColor(gallery_img, cv2.COLOR_RGB2BGR)

            # Save
            save_dir = self.vis_save_dir if self.vis_save_dir else "."
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"neutral_meshes_batch{batch_idx:03d}.png",
            )
            cv2.imwrite(save_path, gallery_img_bgr)
            logger.info(f"Saved neutral meshes gallery: {save_path}")

            # import ipdb; ipdb.set_trace()

        summary_lines = [
            "=" * 60,
            "Average Metrics:",
            "=" * 60,
        ]
        for k, v in metrics.items():
            summary_lines.append(f"{k}: {np.mean(v):.4f}")
        summary_lines.append("=" * 60)

        for line in summary_lines:
            print(line)

        metrics_path = os.path.join(save_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        logger.info(f"Saved metrics summary to {metrics_path}")

        return None

    def merge_predictions(self, mu, sigma):
        """
        Merge predictions from N images
        Args:
            mu: (N, D)
            sigma: (N, D, D)
        Returns:
            mu_star: (D,)
            sigma_star: (D, D)
        """
        inv_sigma = torch.linalg.inv(sigma)

        precision_mat = torch.sum(inv_sigma, dim=0)

        sigma_star = torch.linalg.inv(precision_mat)

        mu_star = sigma_star @ torch.einsum("nij,nj->ni", inv_sigma, mu).sum(dim=0)

        return mu_star, sigma_star

    def merge_predictions_batch(self, mu, sigma):
        """
        Batch version: Merge predictions from N images for B batches
        Args:
            mu: (B, N, D)
            sigma: (B, N, D, D)
        Returns:
            mu_star: (B, D)
            sigma_star: (B, D, D)
        """
        inv_sigma = torch.linalg.inv(sigma)  # (B, N, D, D)

        precision_mat = torch.sum(inv_sigma, dim=1)  # (B, D, D)

        sigma_star = torch.linalg.inv(precision_mat)  # (B, D, D)

        # einsum: bnij,bnj->bni, then sum over N to get (B, D)
        weighted_mu = torch.einsum("bnij,bnj->bni", inv_sigma, mu).sum(dim=1)  # (B, D)

        # Batch matrix multiplication: (B, D, D) @ (B, D) -> (B, D)
        mu_star = torch.bmm(sigma_star, weighted_mu.unsqueeze(-1)).squeeze(-1)  # (B, D)

        return mu_star, sigma_star

    def visualize_keypoints_2d(
        self,
        batch: Dict,
        outputs: Dict,
        batch_idx: int = 0,
        save_path: str = "temp_vis.png",
    ):
        """
        Visualize ground truth and predicted 2D keypoints on the cropped image.

        Args:
            batch: Input batch dictionary
            outputs: Model outputs dictionary
            batch_idx: Batch index to visualize (default: 0)
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt

        # Extract keypoints
        gt_kp2d = (
            batch["keypoints_2d"][batch_idx, :, :].cpu().detach().numpy()
        )  # [N, 2]
        pred_kp2d_cropped_normalised_coords = (
            outputs["mhr"]["pred_keypoints_2d_cropped"][batch_idx]
            .cpu()
            .detach()
            .numpy()
        )  # [70, 2]
        pred_kp2d_cropped_coords = (
            pred_kp2d_cropped_normalised_coords + 0.5
        ) * 256  # [70, 2]

        # Get cropped image
        img = batch["img"][batch_idx, 0].clone().cpu().detach().numpy()  # C, H, W
        img = img.transpose(1, 2, 0)  # H, W, C

        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(
            gt_kp2d[:, 0], gt_kp2d[:, 1], color="blue", s=10, marker="x", label="GT"
        )
        plt.scatter(
            pred_kp2d_cropped_coords[:, 0],
            pred_kp2d_cropped_coords[:, 1],
            color="red",
            s=10,
            marker="o",
            label="Pred Cropped Coords",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_keypoints_2d_samples(
        self,
        batch: Dict,
        outputs: Dict,
        batch_idx: int = 0,
        save_path: str = "temp_vis_samples.png",
    ):
        """
        Visualize ground truth, predicted mean, and sampled 2D keypoints on the cropped image.

        Args:
            batch: Input batch dictionary
            outputs: Model outputs dictionary
            batch_idx: Batch index to visualize (default: 0)
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt

        if "mhr_samples_keypoints_2d" not in outputs:
            logger.warning(
                "No sample keypoints found in outputs. Skipping visualization."
            )
            return

        # Extract keypoints
        gt_kp2d = (
            batch["keypoints_2d"][batch_idx, :, :].cpu().detach().numpy()
        )  # [N, 2]
        pred_kp2d_cropped_normalised_coords = (
            outputs["mhr"]["pred_keypoints_2d_cropped"][batch_idx]
            .cpu()
            .detach()
            .numpy()
        )  # [70, 2]
        pred_kp2d_cropped_coords = (
            pred_kp2d_cropped_normalised_coords + 0.5
        ) * 256  # [70, 2]

        # Extract sample keypoints (in full image coords)
        sample_kp2d_full = (
            outputs["mhr_samples_keypoints_2d"][batch_idx].cpu().detach().numpy()
        )  # [num_samples, 70, 2]

        # Convert samples to cropped coordinates
        num_samples = sample_kp2d_full.shape[0]
        sample_kp2d_cropped_coords = []

        # Ensure body_batch_idx is set (should be set during forward pass)
        if (
            not hasattr(self.model, "body_batch_idx")
            or len(self.model.body_batch_idx) == 0
        ):
            self.model.body_batch_idx = [batch_idx]

        for i in range(num_samples):
            # Convert full image coords to cropped coords using _full_to_crop
            sample_kp2d_tensor = torch.from_numpy(sample_kp2d_full[i : i + 1]).to(
                batch["img"].device
            )
            sample_kp2d_cropped_normalized = (
                self.model._full_to_crop(
                    batch,
                    sample_kp2d_tensor,
                    (
                        torch.tensor(
                            [self.model.body_batch_idx[batch_idx]],
                            device=batch["img"].device,
                        )
                        if len(self.model.body_batch_idx) > batch_idx
                        else None
                    ),
                )[0]
                .cpu()
                .detach()
                .numpy()
            )  # [70, 2] in normalized cropped coords [-0.5, 0.5]

            # Convert to pixel coordinates
            sample_kp2d_cropped = (
                sample_kp2d_cropped_normalized + 0.5
            ) * 256  # [70, 2]
            sample_kp2d_cropped_coords.append(sample_kp2d_cropped)

        sample_kp2d_cropped_coords = np.array(
            sample_kp2d_cropped_coords
        )  # [num_samples, 70, 2]

        # Get cropped image
        img = batch["img"][batch_idx, 0].clone().cpu().detach().numpy()  # C, H, W
        img = img.transpose(1, 2, 0)  # H, W, C

        # Plot samples visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(
            gt_kp2d[:, 0],
            gt_kp2d[:, 1],
            color="blue",
            s=15,
            marker="x",
            label="GT",
            linewidths=2,
        )
        plt.scatter(
            pred_kp2d_cropped_coords[:, 0],
            pred_kp2d_cropped_coords[:, 1],
            color="red",
            s=15,
            marker="o",
            label="Pred Mean",
            linewidths=2,
            edgecolors="darkred",
        )

        # Plot each sample with different colors
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
            )  # Only label first 5 to avoid clutter

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
