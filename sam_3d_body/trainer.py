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
# from .losses.loss import Loss
from .losses.nf_loss import Loss
from .data.bedlam_dataset import DatasetHMR as BEDLAMDataset
from .data.bedlam_dataset import MultiViewEvaluationDataset
from .metrics.metrics_tracker import Metrics
from .visualization.my_vis import Visualiser

# from .configs.config import INDICES_PATH

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

        self.use_lora = cfg.MODEL.DECODER.USE_LORA
        self.model_type = cfg.TRAIN.get("MODEL_TYPE", "full")
        if self.model_type == "toy":
            assert False
            self.model = ToyModel(cfg)
        elif self.model_type == "full":
            self.model = SAM3DBody(cfg)
        else:
            raise ValueError("Invalid model type")

        self.metrics = Metrics()

        # Optionally enable dense keypoints based on config; if disabled, the model
        # will only use the canonical 70 MHR keypoints.
        self.use_dense_keypoints = bool(
            getattr(self.cfg.MODEL, "DENSE_KEYPOINTS", False)
        )
        self.mhr_dense_kp_indices = None
        if self.use_dense_keypoints:
            # mhr_dense_kp_indices_np = np.load(INDICES_PATH)
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
                # Freeze all parameters first
                for param in self.model.parameters():
                    param.requires_grad = False

            # Unfreeze LoRA parameters if LoRA is enabled
            lora_param_count = 0
            if (
                self.use_lora
                and hasattr(self.model, "decoder")
                and hasattr(self.model.decoder, "lora_layers")
            ):
                if self.model.decoder.lora_layers is not None:
                    for lora_layer in self.model.decoder.lora_layers:
                        # LoRA parameters are injected into the base model by PEFT
                        # They typically have names containing "lora_A" and "lora_B"
                        for name, param in lora_layer.named_parameters():
                            if "lora" in name.lower():
                                param.requires_grad = True
                                lora_param_count += param.numel()

            # Unfreeze uncertainty parameters
            for param in [
                self.model.head_uncertainty,
            ]:
                for p in param.parameters():
                    p.requires_grad = True

            # Count and print trainable vs frozen parameters
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            frozen_params = sum(
                p.numel() for p in self.model.parameters() if not p.requires_grad
            )
            decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
            if self.use_lora:
                decoder_lora_params = sum(
                    p.numel() for p in self.model.decoder.lora_layers.parameters()
                )
            total_params = trainable_params + frozen_params

            logger.info("=" * 60)
            logger.info("Parameter Statistics:")
            logger.info("=" * 60)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(
                f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
            )
            logger.info(
                f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)"
            )
            logger.info(
                f"Decoder parameters: {decoder_params:,} ({100 * decoder_params / total_params:.2f}%)"
            )
            if self.use_lora:
                logger.info(
                    f"LoRA decoder parameters: {decoder_lora_params:,} ({100 * decoder_lora_params / total_params:.2f}%)"
                )
                logger.info(f"LoRA trainable parameters: {lora_param_count:,}")
            logger.info("=" * 60)

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
            vis_step = int(
                self.global_step if self.global_step > 0 else (batch_idx or 0)
            )

        should_visualize = vis_step in [0, 250, 500, 1000, 2000, 3000, 4000] or (
            vis_step > 4000 and vis_step % 5000 == 0
        )
        global_rank = getattr(self, "global_rank", 0)
        if should_visualize and global_rank == 0:
        # if global_rank == 0:
            image = batch["img_ori"][0].data  # H W 3, bedlam 720 1280 3
            # image = batch['img'][0,0].data # [3, 256, 256] - CHW format, normalized
            image = image.cpu().detach().numpy()  # [3, H, W]

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
                overlay_gt=True,
                plot_side=True,
                batch=batch,
            )
            # rend_img_samples = my_visualize_samples(
            #     image,
            #     outputs,
            #     self.faces,
            #     stack_vertically=self.stack_vertically,
            # )

            rend_img_bgr = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
            # rend_img_samples_bgr = cv2.cvtColor(rend_img_samples, cv2.COLOR_RGB2BGR)
            rend_img_samples_crops_bgr = cv2.cvtColor(
                rend_img_samples_crops, cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                os.path.join(self.vis_save_dir, f"{vis_step:06d}_img.png"),
                rend_img_bgr,
            )
            # cv2.imwrite(
            #     os.path.join(self.vis_save_dir, f"{vis_step:06d}_samples.png"),
            #     rend_img_samples_bgr,
            # )
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

        kp2d = project(gt_keypoints_3d, trans_cam.unsqueeze(1), cam_int)[:, :70, :2]

        # import matplotlib.pyplot as plt
        # plt.imshow(batch["img_ori"][0].data.cpu().numpy())
        # plt.scatter(verts_2d[0,:, 0].cpu().numpy(), verts_2d[0,:, 1].cpu().numpy(), s=0.1, c='red')
        # plt.title("2D projected mesh vertices")
        # plt.axis("off")
        # plt.savefig("temp_vis.png")
        # plt.close()

        # import ipdb; ipdb.set_trace()

        # Optionally append dense keypoints
        if self.use_dense_keypoints and self.mhr_dense_kp_indices is not None:
            dense_kp2d = project(
                gt_verts[:, self.mhr_dense_kp_indices, :],
                trans_cam.unsqueeze(1),
                cam_int,
            )[:, :, :2]
            kp2d = torch.cat([kp2d, dense_kp2d], dim=1)

        gt_kp2d_h = torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=-1).float()
        affine = batch["affine_trans"][:, 0].float()
        img_size = batch["img_size"][:, 0]

        gt_kp2d_crop = gt_kp2d_h @ affine.mT  # [B, 70, 3] @ [B, 3, 2] = [B, 70, 2]
        # gt_kp2d_crop = gt_kp2d_crop[..., :2]

        gt_kp2d_crop = gt_kp2d_crop / img_size.unsqueeze(1) - 0.5  # [B, 70, 2]
        batch["keypoints_2d"] = gt_kp2d_crop

        # --- temp mirror for joints ---
        j2d = project(gt_joint_coords, trans_cam.unsqueeze(1), cam_int)[..., :2]
        j2d_h = torch.cat([j2d, torch.ones_like(j2d[..., :1])], dim=-1).float()
        j2d_crop = j2d_h @ affine.mT
        j2d_crop = j2d_crop[..., :2]
        j2d_crop = j2d_crop / img_size.unsqueeze(1) - 0.5
        batch["joints_2d"] = j2d_crop

        # ------------ gt for no glob rot ------------
        model_parameters = batch["model_params"]

        # No global transl
        model_parameters[:, :3] = 0

        global_rot = batch["model_params"][:, 3:6]

        global_rotmat = roma.euler_to_rotmat("xyz", global_rot)  # B x 3 x 3

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
        new_global_rotmat = torch.bmm(rot_180_x, global_rotmat)

        global_rot = roma.rotmat_to_euler("xyz", new_global_rotmat)

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
            shuffle=True,  # self.cfg.DATASET.SHUFFLE_TRAIN,
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
            shuffle=False,
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
                "00122",
                "00123",
                "00127",
                "00129",
                "00134",
                "00135",
                "00136",
                "00137",
                "00140",
                "00147",
                "00148",
                "00149",
                "00151",
                "00152",
                "00154",
                "00156",
                "00160",
                "00163",
                "00167",
                "00168",
                "00169",
                "00170",
                "00174",
                "00175",
                "00176",
                "00179",
                "00180",
                "00185",
                "00187",
                "00190",
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

        all_metrics = defaultdict(list)

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

            # Uncertainties come from uncertainty head (separate from mhr output)
            indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
            shape_var = outputs["uncertainty_output"]["shape_uncertainty"]
            scale_var = outputs["uncertainty_output"]["scale_uncertainty"]

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

            scale_mu_star, scale_sigma_star = self.merge_predictions_batch(
                pred_scale[..., indices], scale_var_diag
            )
            scale_mu_star_full = pred_scale.mean(dim=1)
            scale_mu_star_full[..., indices] = scale_mu_star

            shape_mean = pred_shape.mean(dim=1).repeat_interleave(
                num_views, dim=0
            )  # naive average of parameters
            scale_mean = pred_scale.mean(dim=1).repeat_interleave(num_views, dim=0)

            mhr_zero_inputs = {
                "global_trans": torch.zeros_like(outputs["mhr"]["global_rot"]),
                "global_rot": torch.zeros_like(outputs["mhr"]["global_rot"]),
                "body_pose_params": torch.zeros_like(outputs["mhr"]["body_pose"]),
                # "hand_pose_params": torch.zeros_like(outputs["mhr"]["hand"]),
                "expr_params": torch.zeros_like(outputs["mhr"]["face"]),
            }
            mhr_output_config = {
                "return_keypoints": True,
                "return_joint_coords": True,
                "return_model_params": True,
                "return_joint_rotations": True,
                "do_pcblend": True,
            }

            mean_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mean,
                scale_params=scale_mean,
                global_trans=torch.zeros_like(outputs["mhr"]["global_rot"]),
                global_rot=outputs["mhr"]["global_rot"],
                body_pose_params=outputs["mhr"]["body_pose"],
                hand_pose_params=outputs["mhr"]["hand"],
                expr_params=outputs["mhr"]["face"],
                **mhr_output_config
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
                **mhr_output_config
            )
            verts_star, j3d_star, _, _, _ = (
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
                **mhr_zero_inputs,
                **mhr_output_config
            )
            per_view_neutral_verts, _, per_view_neutral_joint_coords, _, _ = (
                per_view_neutral_mhr_output
            )

            # get merged neutral pred
            merged_neutral_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mu_star.repeat_interleave(num_views, dim=0),
                scale_params=scale_mu_star_full.repeat_interleave(num_views, dim=0),
                **mhr_zero_inputs,
                **mhr_output_config
            )
            merged_neutral_verts, _, merged_neutral_joint_coords, _, _ = (
                merged_neutral_mhr_output
            )

            mean_neutral_mhr_output = self.model.head_pose.mhr_forward(
                shape_params=shape_mean,
                scale_params=scale_mean,
                **mhr_zero_inputs,
                **mhr_output_config
            )
            mean_neutral_verts, _, mean_neutral_joint_coords, _, _ = mean_neutral_mhr_output

            stuff_for_metrics = {
                "per_view_neutral_joint_coords": per_view_neutral_joint_coords,
                "merged_neutral_joint_coords": merged_neutral_joint_coords,
                "mean_neutral_joint_coords": mean_neutral_joint_coords,
                "gt_neutral_joint_coords": gt_neutral_joint_coords,
                "per_view_neutral_verts": per_view_neutral_verts,
                "merged_neutral_verts": merged_neutral_verts,
                "mean_neutral_verts": mean_neutral_verts,
                "gt_neutral_verts": gt_neutral_verts,
            }

            self.multiframe_metrics(all_metrics, stuff_for_metrics)



            renderer = Renderer(
                focal_length=outputs["mhr"]["focal_length"][0], faces=self.faces
            )

            stuff_for_vis = {
                "renderer": renderer,
                "outputs": outputs,
                "batch": batch,
                "verts_star": verts_star,
                "metrics": all_metrics,
                "pred_shape": pred_shape,
                "shape_mu_star": shape_mu_star,
                "merged_shape_var": merged_shape_var,
                "shape_var_unflattened": shape_var_unflattened,
                "num_views": num_views,
                "bs": bs,
                "batch_idx": batch_idx,
                "per_view_neutral_joint_coords": per_view_neutral_joint_coords,
                "merged_neutral_joint_coords": merged_neutral_joint_coords,
                "mean_neutral_joint_coords": mean_neutral_joint_coords,
                "gt_neutral_joint_coords": gt_neutral_joint_coords,
                "per_view_neutral_verts": per_view_neutral_verts,
                "merged_neutral_verts": merged_neutral_verts,
                "mean_neutral_verts": mean_neutral_verts,
                "gt_neutral_verts": gt_neutral_verts,
            }
            self.vis_predictions(stuff_for_vis, sc=True)
            self.vis_predictions(stuff_for_vis, sc=False)

            neutral_renderer = Renderer(focal_length=512, faces=self.faces)
            stuff_for_vis["neutral_renderer"] = neutral_renderer

            self.vis_neutral(stuff_for_vis, sc=True)
            self.vis_neutral(stuff_for_vis, sc=False)

        mean_metrics = {}
        for k, v in all_metrics.items():
            try:
                mean_metrics[k] = torch.stack(v).mean().item()
            except:
                mean_metrics[k] = np.mean(np.array(v))

        summary_lines = [
            "=" * 60,
            "Average Metrics:",
            "=" * 60,
        ]

        # print(all_metrics)
        # for k, v in all_metrics.items():
        #     print(f"{k}: {type(v)}")
        # import ipdb; ipdb.set_trace()


        for k, v in mean_metrics.items():
            summary_lines.append(f"{k}: {v:.4f}")
        summary_lines.append("=" * 60)

        for line in summary_lines:
            print(line)

        save_dir = self.vis_save_dir

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

    def multiframe_metrics(self, all_metrics, stuff_for_metrics):
        per_view_neutral_joints = stuff_for_metrics["per_view_neutral_joint_coords"]
        merged_neutral_joints = stuff_for_metrics["merged_neutral_joint_coords"]
        mean_neutral_joints = stuff_for_metrics["mean_neutral_joint_coords"]
        gt_neutral_joints = stuff_for_metrics["gt_neutral_joint_coords"]

        per_view_neutral_verts = stuff_for_metrics["per_view_neutral_verts"]
        merged_neutral_verts = stuff_for_metrics["merged_neutral_verts"]
        mean_neutral_verts = stuff_for_metrics["mean_neutral_verts"]
        gt_neutral_verts = stuff_for_metrics["gt_neutral_verts"]
        # ----- mpjpe -----
        # per-view
        per_view_mpjpe = torch.sqrt(
            ((per_view_neutral_joints - gt_neutral_joints) ** 2).sum(dim=-1)
        ).mean(dim=1)
        # merged
        merged_mpjpe = torch.sqrt(
            ((merged_neutral_joints - gt_neutral_joints) ** 2).sum(dim=-1)
        ).mean(dim=1)
        # mean
        mean_mpjpe = torch.sqrt(
            ((mean_neutral_joints - gt_neutral_joints) ** 2).sum(dim=-1)
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
            per_view_neutral_joints.cpu().detach().numpy(),
            gt_neutral_joints.cpu().detach().numpy(),
            reduction="none",
        )
        per_view_pampjpe = per_view_pampjpe.mean(axis=-1)
        # merged
        merged_pampjpe, _ = reconstruction_error(
            merged_neutral_joints.cpu().detach().numpy(),
            gt_neutral_joints.cpu().detach().numpy(),
            reduction="none",
        )
        merged_pampjpe = merged_pampjpe.mean(axis=-1)
        # mean
        mean_pampjpe, _ = reconstruction_error(
            mean_neutral_joints.cpu().detach().numpy(),
            gt_neutral_joints.cpu().detach().numpy(),
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

        all_metrics["per_view_mpjpe"].append(per_view_mpjpe)
        all_metrics["best_per_view_mpjpe"].append(per_view_mpjpe.min().item())
        all_metrics["mean_mpjpe"].append(mean_mpjpe)
        all_metrics["merged_mpjpe"].append(merged_mpjpe)

        all_metrics["per_view_pve"].append(per_view_pve)
        all_metrics["best_per_view_pve"].append(per_view_pve.min().item())
        all_metrics["mean_pve"].append(mean_pve)
        all_metrics["merged_pve"].append(merged_pve)

        all_metrics["per_view_pampjpe"].append(per_view_pampjpe)
        all_metrics["best_per_view_pampjpe"].append(per_view_pampjpe.min().item())
        all_metrics["mean_pampjpe"].append(mean_pampjpe)
        all_metrics["merged_pampjpe"].append(merged_pampjpe)

        all_metrics["per_view_pvetsc"].append(per_view_pvetsc)
        all_metrics["best_per_view_pvetsc"].append(per_view_pvetsc.min().item())
        all_metrics["mean_pvetsc"].append(mean_pvetsc)
        all_metrics["merged_pvetsc"].append(merged_pvetsc)

        return all_metrics


    def vis_predictions(
        self,
        input_dict,
        plot_side: bool = True,
        overlay_sideview: bool = True,
        sc: bool = True,
    ):
        renderer = input_dict["renderer"]
        outputs = input_dict["outputs"]
        batch = input_dict["batch"]
        verts_star = input_dict["verts_star"]
        num_views = input_dict["num_views"]
        bs = input_dict["bs"]
        batch_idx = input_dict["batch_idx"]
        pred_shape = input_dict["pred_shape"]
        shape_var_unflattened = input_dict["shape_var_unflattened"]
        shape_mu_star = input_dict["shape_mu_star"]
        merged_shape_var = input_dict["merged_shape_var"]
        metrics = input_dict["metrics"]

        # Initialize gallery: list of lists to store rendered images [bs][num_views]
        gallery = [[None for _ in range(num_views)] for _ in range(bs)]

        i=0
        all_distances = []
        pred_vertex_dists = {}
        merged_vertex_dists = {}

        for view in range(num_views):
            flat_idx = i * num_views + view

            # verts = (
            #     outputs["mhr"]["pred_vertices"][flat_idx].cpu().detach().numpy()
            # )
            # gt_verts = (
            #     batch["gt_verts_w_transl"][flat_idx].cpu().detach().numpy()
            # )
            # merged_verts = verts_star[flat_idx].cpu().detach().numpy()

            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D

            # fig = plt.figure(figsize=(8, 6))
            # ax = fig.add_subplot(111, projection='3d')

            # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='b', s=1, label='Pred. Verts')
            # ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], c='r', s=1, label='GT Verts')

            # ax.set_title('3D Scatter: Predicted vs Ground Truth Vertices')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.legend()
            # plt.savefig('pred_vs_gt_verts.png')
            # plt.close()
            # import ipdb; ipdb.set_trace()

            verts = input_dict["per_view_neutral_verts"][flat_idx].cpu().detach().numpy()
            gt_verts = input_dict["gt_neutral_verts"][flat_idx].cpu().detach().numpy()
            merged_verts = input_dict["merged_neutral_verts"][flat_idx].cpu().detach().numpy()

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

            verts = outputs["mhr"]["pred_vertices"][flat_idx].cpu().detach().numpy()
            cam_t = outputs["mhr"]["pred_cam_t"][flat_idx].cpu().detach().numpy()

            gt_verts = batch["gt_verts_w_transl"][flat_idx].cpu().detach().numpy()
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
                    mesh_base_color=(1.0, 0.8, 0.5),
                    scene_bg_color=(1, 1, 1),
                    camera_center=(
                        batch["cam_int"][flat_idx][0, 2],
                        batch["cam_int"][flat_idx][1, 2],
                    ),
                    vertex_colors=pred_colors,
                )
                * 255
            ).astype(np.uint8)

            # Overlay semi-transparent GT mesh on top of predicted mesh (light orange)
            gt_rgba = renderer(
                gt_verts,
                gt_cam_t,
                np.ones_like(img_for_render) * 255,
                mesh_base_color=(1.0, 0.8, 0.5),
                scene_bg_color=(1, 1, 1),
                camera_center=(
                    batch["cam_int"][flat_idx][0, 2],
                    batch["cam_int"][flat_idx][1, 2],
                ),
                return_rgba=True,
            )
            alpha = (gt_rgba[..., 3:4].astype(np.float32) * 0.5)
            pred_rgb = rendered_img.astype(np.float32) / 255.0
            gt_rgb = gt_rgba[..., :3].astype(np.float32)
            blended_pred = alpha * gt_rgb + (1.0 - alpha) * pred_rgb
            rendered_img = (blended_pred * 255.0).clip(0, 255).astype(np.uint8)

            # Merged mesh: per-vertex viridis colors from distance to GT
            merged_colors = build_vertex_colors(merged_vertex_dists[view])
            rendered_merged_img = (
                renderer(
                    merged_verts,
                    cam_t,
                    img_for_render.copy(),
                    mesh_base_color=(0.5, 1.0, 0.5),
                    scene_bg_color=(1, 1, 1),
                    camera_center=(
                        batch["cam_int"][flat_idx][0, 2],
                        batch["cam_int"][flat_idx][1, 2],
                    ),
                    vertex_colors=merged_colors,
                )
                * 255
            ).astype(np.uint8)

            # Overlay semi-transparent GT mesh on top of merged mesh (light orange)
            gt_rgba_merged = renderer(
                gt_verts,
                gt_cam_t,
                np.ones_like(img_for_render) * 255,
                mesh_base_color=(1.0, 0.8, 0.5),
                scene_bg_color=(1, 1, 1),
                camera_center=(
                    batch["cam_int"][flat_idx][0, 2],
                    batch["cam_int"][flat_idx][1, 2],
                ),
                return_rgba=True,
            )
            alpha_m = (gt_rgba_merged[..., 3:4].astype(np.float32) * 0.3)
            merged_rgb = rendered_merged_img.astype(np.float32) / 255.0
            gt_m_rgb = gt_rgba_merged[..., :3].astype(np.float32)
            blended_merged = alpha_m * gt_m_rgb + (1.0 - alpha_m) * merged_rgb
            rendered_merged_img = (blended_merged * 255.0).clip(0, 255).astype(np.uint8)

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
            per_view_pampjpe = metrics["per_view_pampjpe"][-1]
            merged_pampjpe = metrics["merged_pampjpe"][-1]
            per_view_pvetsc = metrics["per_view_pvetsc"][-1]
            merged_pvetsc = metrics["merged_pvetsc"][-1]
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

            if plot_side:
                # Create side-view renders (90-degree rotation around Y) on white background
                white_bg = np.ones_like(img_for_render) * 255

                if overlay_sideview:
                    # Center meshes at origin to align GT and predictions, then render with a generic camera
                    generic_cam_t = np.array([0.0, -0.25, 2.5])

                    # Optionally scale-normalize prediction and merged meshes to GT (similar to multiframe_metrics)
                    if sc:
                        from sam_3d_body.metrics.metrics_tracker import (
                            scale_and_translation_transform_batch,
                        )

                        verts_sn = scale_and_translation_transform_batch(
                            verts[None, ...], gt_verts[None, ...]
                        )[0]
                        merged_sn = scale_and_translation_transform_batch(
                            merged_verts[None, ...], gt_verts[None, ...]
                        )[0]

                        # Use a common center (from GT) so GT and predictions are aligned in side view
                        center = gt_verts.mean(axis=0, keepdims=True)
                        gt_centered = gt_verts - center
                        verts_centered = verts_sn - center
                        merged_centered = merged_sn - center
                    else:
                        verts_sn = verts
                        merged_sn = merged_verts

                        gt_centered = gt_verts - gt_verts.mean(axis=0, keepdims=True)
                        verts_centered = verts - verts.mean(axis=0, keepdims=True)
                        merged_centered = merged_verts - merged_verts.mean(axis=0, keepdims=True)


                    # GT side view (for the GT column)
                    gt_side = (
                        renderer(
                            gt_centered,
                            generic_cam_t,
                            white_bg.copy(),
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            side_view=True,
                            rot_angle=90,
                        )
                        * 255
                    ).astype(np.uint8)

                    # Pred side view with GT overlay
                    pred_side_base = (
                        renderer(
                            verts_centered,
                            generic_cam_t,
                            white_bg.copy(),
                            mesh_base_color=(1.0, 0.8, 0.5),
                            scene_bg_color=(1, 1, 1),
                            vertex_colors=pred_colors,
                            side_view=True,
                            rot_angle=90,
                        )
                        * 255
                    ).astype(np.uint8)
                    gt_side_rgba = renderer(
                        gt_centered,
                        generic_cam_t,
                        white_bg.copy(),
                        mesh_base_color=(1.0, 0.8, 0.5),
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                        rot_angle=90,
                        return_rgba=True,
                    )
                    alpha_side = (gt_side_rgba[..., 3:4].astype(np.float32) * 0.5)
                    pred_side_rgb = pred_side_base.astype(np.float32) / 255.0
                    gt_side_rgb = gt_side_rgba[..., :3].astype(np.float32)
                    blended_pred_side = alpha_side * gt_side_rgb + (1.0 - alpha_side) * pred_side_rgb
                    pred_side = (blended_pred_side * 255.0).clip(0, 255).astype(np.uint8)

                    # Merged side view with GT overlay
                    merged_side_base = (
                        renderer(
                            merged_centered,
                            generic_cam_t,
                            white_bg.copy(),
                            mesh_base_color=(0.5, 1.0, 0.5),
                            scene_bg_color=(1, 1, 1),
                            vertex_colors=merged_colors,
                            side_view=True,
                            rot_angle=90,
                        )
                        * 255
                    ).astype(np.uint8)
                    merged_side_rgb = merged_side_base.astype(np.float32) / 255.0
                    blended_merged_side = alpha_side * gt_side_rgb + (1.0 - alpha_side) * merged_side_rgb
                    merged_side = (blended_merged_side * 255.0).clip(0, 255).astype(np.uint8)

                else:
                    # Original behavior: independent side views using view-specific camera
                    gt_side = (
                        renderer(
                            gt_verts,
                            gt_cam_t,
                            white_bg.copy(),
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            camera_center=(
                                batch["cam_int"][flat_idx][0, 2],
                                batch["cam_int"][flat_idx][1, 2],
                            ),
                            side_view=True,
                            rot_angle=90,
                        )
                        * 255
                    ).astype(np.uint8)

                    pred_side = (
                        renderer(
                            verts,
                            cam_t,
                            white_bg.copy(),
                            mesh_base_color=(1.0, 0.8, 0.5),
                            scene_bg_color=(1, 1, 1),
                            camera_center=(
                                batch["cam_int"][flat_idx][0, 2],
                                batch["cam_int"][flat_idx][1, 2],
                            ),
                            vertex_colors=pred_colors,
                            side_view=True,
                            rot_angle=90,
                        )
                        * 255
                    ).astype(np.uint8)

                    merged_side = (
                        renderer(
                            merged_verts,
                            cam_t,
                            white_bg.copy(),
                            mesh_base_color=(0.5, 1.0, 0.5),
                            scene_bg_color=(1, 1, 1),
                            camera_center=(
                                batch["cam_int"][flat_idx][0, 2],
                                batch["cam_int"][flat_idx][1, 2],
                            ),
                            vertex_colors=merged_colors,
                            side_view=True,
                            rot_angle=90,
                        )
                        * 255
                    ).astype(np.uint8)

                gallery[i][view] = np.concatenate(
                    [
                        gt_rendered_img,
                        gt_side,
                        rendered_img,
                        pred_side,
                        rendered_merged_img,
                        merged_side,
                    ],
                    axis=1,
                )
            else:
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
        # Downscale final image by factor 2 before saving
        h, w = gallery_img_bgr.shape[:2]
        gallery_img_bgr = cv2.resize(
            gallery_img_bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA
        )

        save_dir = self.vis_save_dir if self.vis_save_dir else "."
        os.makedirs(save_dir, exist_ok=True)
        suffix = "_sc" if sc else ""
        save_path = os.path.join(
            save_dir,
            # f"batch{batch_idx:03d}_bs{bs}_views{num_views}{suffix}.png",
            f"b{batch_idx:03d}{suffix}.png",
        )
        cv2.imwrite(save_path, gallery_img_bgr)
        logger.info(
            f"Saved multiview gallery: {save_path} (shape: {gallery_img.shape})"
        )


    def vis_neutral(self, stuff_for_vis, sc: bool = True):
        neutral_renderer = stuff_for_vis["neutral_renderer"]
        batch = stuff_for_vis["batch"]
        gt_neutral_verts = stuff_for_vis["gt_neutral_verts"]
        merged_neutral_verts = stuff_for_vis["merged_neutral_verts"]
        per_view_neutral_verts = stuff_for_vis["per_view_neutral_verts"]
        num_views = stuff_for_vis["num_views"]
        batch_idx = stuff_for_vis["batch_idx"]
        bs = stuff_for_vis["bs"]

        generic_cam_t = np.array([0.0, 0.75, 2.5])

        # ----- Prepare vertices (optionally scale-normalized) -----
        # Work in the canonical (unflipped) coordinate frame for distances.
        gt_neutral_verts_np = gt_neutral_verts.cpu().detach().numpy()
        per_view_verts_np = per_view_neutral_verts.cpu().detach().numpy()
        merged_verts_np = merged_neutral_verts.cpu().detach().numpy()

        # When sc=True, scale-normalize per-view and merged meshes to GT, similar to multiframe_metrics.
        if sc:
            from sam_3d_body.metrics.metrics_tracker import (
                scale_and_translation_transform_batch,
            )

            per_view_verts_np = scale_and_translation_transform_batch(
                per_view_verts_np, gt_neutral_verts_np
            )
            merged_verts_np = scale_and_translation_transform_batch(
                merged_verts_np, gt_neutral_verts_np
            )

        # Use the last GT mesh as reference (consistent with previous vis2 behavior).
        gt_ref = gt_neutral_verts_np[-1]

        all_distances = []
        per_view_vertex_dists = {}

        for view in range(num_views):
            pv_verts = per_view_verts_np[view]
            dist_pv = np.linalg.norm(pv_verts - gt_ref, axis=1)
            per_view_vertex_dists[view] = dist_pv
            all_distances.append(dist_pv)

        merged_verts_ref = merged_verts_np[0]
        merged_vertex_dists = np.linalg.norm(merged_verts_ref - gt_ref, axis=1)
        all_distances.append(merged_vertex_dists)

        all_distances = np.concatenate(all_distances)
        min_dist = float(all_distances.min()) if all_distances.size > 0 else 0.0
        max_dist = float(all_distances.max()) if all_distances.size > 0 else 1.0

        if max_dist > min_dist:
            denom = max_dist - min_dist
        else:
            denom = 1.0

        def build_vertex_colors(dists: np.ndarray) -> np.ndarray:
            """Map distances to RGBA vertex colors using shared viridis scale (as in vis1)."""
            normalized = (dists - min_dist) / denom
            normalized = np.clip(normalized, 0.0, 1.0)
            colors_rgb = plt.cm.viridis(normalized)[..., :3]  # (V, 3)
            vertex_colors = np.ones((colors_rgb.shape[0], 4), dtype=np.float32)
            vertex_colors[:, :3] = colors_rgb
            return vertex_colors

        # ----- Render GT neutral mesh (solid color) -----
        gt_neutral_verts_vis = gt_neutral_verts_np.copy()
        gt_neutral_verts_vis[..., [1, 2]] *= -1
        gt_neutral_rendered = (
            neutral_renderer(
                gt_neutral_verts_vis[-1],
                generic_cam_t,
                np.ones((512, 512, 3)) * 255,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        (text_width, text_height), baseline = cv2.getTextSize(
            "GT", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )

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

        # Side view of GT neutral mesh on white background
        white_bg = np.ones((512, 512, 3)) * 255
        gt_neutral_side = (
            neutral_renderer(
                gt_neutral_verts_vis[-1],
                generic_cam_t,
                white_bg.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
                rot_angle=90,
            )
            * 255
        ).astype(np.uint8)

        # Side-view GT overlay (light orange) for second-row views
        gt_neutral_side_rgba = neutral_renderer(
            gt_neutral_verts_vis[-1],
            generic_cam_t,
            white_bg.copy(),
            mesh_base_color=(1.0, 0.8, 0.5),
            scene_bg_color=(1, 1, 1),
            side_view=True,
            rot_angle=90,
            return_rgba=True,
        )
        gt_side_alpha = (gt_neutral_side_rgba[..., 3:4].astype(np.float32) * 0.5)
        gt_side_rgb = gt_neutral_side_rgba[..., :3].astype(np.float32)

        # Pre-compute semi-transparent GT overlay (RGBA) for neutral views (light orange)
        gt_neutral_rgba = neutral_renderer(
            gt_neutral_verts_vis[-1],
            generic_cam_t,
            np.ones((512, 512, 3)) * 255,
            mesh_base_color=(1.0, 0.8, 0.5),
            scene_bg_color=(1, 1, 1),
            return_rgba=True,
        )
        gt_alpha = (gt_neutral_rgba[..., 3:4].astype(np.float32) * 0.5)
        gt_rgb = gt_neutral_rgba[..., :3].astype(np.float32)

        # ----- Render per-view neutral meshes with per-vertex colors -----
        per_view_rendered_front = []
        per_view_rendered_side = []
        per_view_verts_vis = per_view_verts_np.copy()
        per_view_verts_vis[..., [1, 2]] *= -1

        for view in range(num_views):
            vertex_colors = build_vertex_colors(per_view_vertex_dists[view])
            rendered = (
                neutral_renderer(
                    per_view_verts_vis[view],
                    generic_cam_t,
                    np.ones((512, 512, 3)) * 255,
                    mesh_base_color=(1.0, 0.8, 0.5),  # unused when vertex_colors is set
                    scene_bg_color=(1, 1, 1),
                    vertex_colors=vertex_colors,
                )
                * 255
            ).astype(np.uint8)
            per_view_rendered_front.append(rendered)

            # Overlay semi-transparent GT neutral mesh on top
            pv_rgb = rendered.astype(np.float32) / 255.0
            blended_pv = gt_alpha * gt_rgb + (1.0 - gt_alpha) * pv_rgb
            rendered = (blended_pv * 255.0).clip(0, 255).astype(np.uint8)
            per_view_rendered_front[-1] = rendered

            cv2.putText(
                rendered,
                f"View {view}",
                **text_config,
            )

            # Side view for this per-view neutral mesh (with GT overlay, white background)
            rendered_side = (
                neutral_renderer(
                    per_view_verts_vis[view],
                    generic_cam_t,
                    white_bg.copy(),
                    mesh_base_color=(1.0, 0.8, 0.5),
                    scene_bg_color=(1, 1, 1),
                    vertex_colors=vertex_colors,
                    side_view=True,
                    rot_angle=90,
                )
                * 255
            ).astype(np.uint8)
            pv_side_rgb = rendered_side.astype(np.float32) / 255.0
            blended_side = gt_side_alpha * gt_side_rgb + (1.0 - gt_side_alpha) * pv_side_rgb
            rendered_side = (blended_side * 255.0).clip(0, 255).astype(np.uint8)
            per_view_rendered_side.append(rendered_side)

        # ----- Render merged neutral mesh with per-vertex colors -----
        merged_verts_vis = merged_verts_ref.copy()
        merged_verts_vis[..., [1, 2]] *= -1
        merged_vertex_colors = build_vertex_colors(merged_vertex_dists)
        merged_neutral_rendered = (
            neutral_renderer(
                merged_verts_vis,
                generic_cam_t,
                np.ones((512, 512, 3)) * 255,
                mesh_base_color=(0.5, 1.0, 0.5),  # unused when vertex_colors is set
                scene_bg_color=(1, 1, 1),
                vertex_colors=merged_vertex_colors,
            )
            * 255
        ).astype(np.uint8)

        # Overlay semi-transparent GT neutral mesh on top of merged neutral mesh
        merged_rgb = merged_neutral_rendered.astype(np.float32) / 255.0
        blended_merged = gt_alpha * gt_rgb + (1.0 - gt_alpha) * merged_rgb
        merged_neutral_rendered = (blended_merged * 255.0).clip(0, 255).astype(np.uint8)

        cv2.putText(
            merged_neutral_rendered,
            "Merged",
            **text_config,
        )

        # Side view for merged neutral mesh (with GT overlay, white background)
        merged_neutral_side = (
            neutral_renderer(
                merged_verts_vis,
                generic_cam_t,
                white_bg.copy(),
                mesh_base_color=(0.5, 1.0, 0.5),
                scene_bg_color=(1, 1, 1),
                vertex_colors=merged_vertex_colors,
                side_view=True,
                rot_angle=90,
            )
            * 255
        ).astype(np.uint8)
        merged_side_rgb = merged_neutral_side.astype(np.float32) / 255.0
        blended_merged_side = gt_side_alpha * gt_side_rgb + (1.0 - gt_side_alpha) * merged_side_rgb
        merged_neutral_side = (blended_merged_side * 255.0).clip(0, 255).astype(np.uint8)

        # Assemble gallery into three rows:
        #   top    = front views (GT, merged, per-view)
        #   middle = side views (GT, merged, per-view)
        #   bottom = original images per view (GT / merged columns left blank)
        top_row_images = [gt_neutral_rendered, merged_neutral_rendered]
        bottom_row_images = [gt_neutral_side, merged_neutral_side]
        for view in range(num_views):
            top_row_images.append(per_view_rendered_front[view])
            bottom_row_images.append(per_view_rendered_side[view])

        top_row = np.concatenate(top_row_images, axis=1)
        bottom_row = np.concatenate(bottom_row_images, axis=1)

        # Third row: blank for GT and merged, then per-view input images
        # Keep original aspect ratio for input images, but align widths with first two rows
        tile_h, tile_w = gt_neutral_rendered.shape[:2]
        blank = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255
        # third_row_images = [blank.copy(), blank.copy()]

        # for view in range(num_views):
        #     # Use cropped input image instead of full image
        #     img_t = batch["img"][view].cpu().detach()
        #     img = img_t.numpy()
        #     # Convert from CHW to HWC if needed
        #     if img.ndim == 3 and img.shape[0] == 3:
        #         img = np.transpose(img, (1, 2, 0))
        #     # Denormalize to [0, 255] and convert to uint8
        #     img = (img * 255.0).clip(0, 255).astype(np.uint8)

        #     ih, iw = img.shape[:2]
        #     # Compute scale to fit inside (tile_h, tile_w) while preserving aspect ratio
        #     scale = min(tile_w / iw, tile_h / ih)
        #     new_w = max(1, int(round(iw * scale)))
        #     new_h = max(1, int(round(ih * scale)))

        #     img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        #     # Paste resized image into centered white tile
        #     canvas = blank.copy()
        #     y_off = (tile_h - new_h) // 2
        #     x_off = (tile_w - new_w) // 2
        #     canvas[y_off : y_off + new_h, x_off : x_off + new_w] = img_resized
        #     third_row_images.append(canvas)

        # third_row = np.concatenate(third_row_images, axis=1)

        gallery_img = np.concatenate([top_row, bottom_row], axis=0)
        gallery_img_bgr = cv2.cvtColor(gallery_img, cv2.COLOR_RGB2BGR)
        # Downscale final image by factor 2 before saving
        # h, w = gallery_img_bgr.shape[:2]
        # gallery_img_bgr = cv2.resize(
        #     gallery_img_bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA
        # )

        # Save
        save_dir = self.vis_save_dir if self.vis_save_dir else "."
        os.makedirs(save_dir, exist_ok=True)
        suffix = "_sc" if sc else ""
        save_path = os.path.join(
            save_dir,
            f"b{batch_idx:03d}_neutral{suffix}.png",
        )
        cv2.imwrite(save_path, gallery_img_bgr)
        logger.info(f"Saved neutral meshes gallery: {save_path}")