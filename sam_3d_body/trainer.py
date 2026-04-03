import os
import cv2
import roma
import torch
import numpy as np

from typing import Dict, Optional
from collections import defaultdict
from yacs.config import CfgNode
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader

from .models.meta_arch.sam3d_body import SAM3DBody
from .models.meta_arch.base_lightning_module import BaseLightningModule
from .models.meta_arch.nf_merging import get_mhr_outputs
from .models.meta_arch.nf_merging import merge_params_nf

# from .losses.loss import Loss
from .losses.nf_loss import Loss
from .data.bedlam_dataset import DatasetHMR as BEDLAMDataset
from .data.bedlam_dataset import bedlam_collate
from .data.bedlam_dataset import MultiViewEvaluationDataset
from .metrics.metrics_tracker import Metrics
from .metrics.metrics_tracker import (
    multiframe_metrics,
    print_multiview_metrics,
    scale_and_translation_transform_batch,
)
from .visualization.my_vis import Visualiser, vis_predictions, vis_neutral
from .visualization.renderer import Renderer

from .configs.config import INDICES_PATH

import sys
from pathlib import Path

# Add project root to path for tools import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.vis_utils import my_visualize
from tools.vis_utils import my_visualize_samples
from tools.vis_utils import view_one_in_another


def _write_obj(path, verts, faces):
    """Write a single mesh to a Wavefront OBJ file (faces are 0-indexed numpy arrays)."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def export_meshes_for_blender(outs, faces, save_dir, tag="b002"):
    """
    Save GT, merged, and per-view-mean-prediction neutral meshes as OBJ files.
    All prediction meshes are scale-normalised (scale+translation) to match the GT mesh.
    GT is saved in its original scale as the reference frame.
    Files are written to save_dir with names: <tag>_gt.obj, <tag>_merged.obj, <tag>_view00.obj, ...
    """
    gt_np = outs["gt_neutral_verts"].cpu().numpy()          # [B*V, n_verts, 3]
    pred_np = outs["per_view_neutral_verts"].cpu().numpy()  # [B*V, n_verts, 3]
    merged_np = outs["merged_neutral_verts"].cpu().numpy()  # [B*V, n_verts, 3]

    pred_sc = scale_and_translation_transform_batch(pred_np, gt_np)
    merged_sc = scale_and_translation_transform_batch(merged_np, gt_np)

    # GT: one mesh, same for all views — save view 0
    _write_obj(os.path.join(save_dir, f"{tag}_gt.obj"), gt_np[0], faces)
    # Merged: one mesh, same for all views — save view 0
    _write_obj(os.path.join(save_dir, f"{tag}_merged.obj"), merged_sc[0], faces)
    # Per-view mean predictions
    for i in range(pred_sc.shape[0]):
        _write_obj(os.path.join(save_dir, f"{tag}_view{i:02d}.obj"), pred_sc[i], faces)

    print(f"Saved Blender meshes to {save_dir} (tag={tag})")


class Trainer(BaseLightningModule):
    """
    Trainer class that extends SAM3DBody with PyTorch Lightning training logic.
    Inherits all model functionality from SAM3DBody.
    """

    def __init__(
        self,
        cfg: CfgNode,
        vis_save_dir: str = None,
        stack_vertically: bool = True,
        always_visualise: bool = False,
    ):
        super().__init__()

        self.cfg = cfg
        self.vis_save_dir = vis_save_dir
        self.stack_vertically = stack_vertically
        self.always_visualise = always_visualise

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
                # Freeze all parameters first
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                assert False 

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
                # self.model.head_uncertainty,
                self.model.nf_head,
            ]:
                for p in param.parameters():
                    p.requires_grad = True

        self.scale_mean = self.model.head_pose.scale_mean.float()
        self.scale_comps = self.model.head_pose.scale_comps.float()

        self.criterion = Loss(
            cfg,
            scale_mean=self.scale_mean,
            scale_comps=self.scale_comps,
            nf_head=self.model.nf_head,
        )

        self.faces = self.model.head_pose.faces.cpu().detach().numpy()

        self.visualiser = Visualiser(vis_save_dir, cfg=cfg, faces=self.faces)

    def training_step(self, batch: Dict, batch_idx: int):
        
        batch = self.preprocess(batch)

        outputs = self(batch, num_samples=self.cfg.MODEL.NUM_SAMPLES)

        loss_dict = self.criterion(outputs, batch)

        metrics = self.metrics(outputs, batch)

        self.log_and_visualise(
            loss_dict, metrics, batch, outputs, prefix="train_", batch_idx=batch_idx
        )

        for k, v in loss_dict.items():
            print(f"{k}: {v.item():.3f}", end=" ")
        print("")
        # for k, v in metrics.items():
        #     print(f"{k}: {v:.4f}", end=" ")
        # print("")
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

        raw_metrics = metrics.copy()
        # Also propagate selected loss quantities needed for visualization (e.g., GT log-prob).
        if "gt_residual_log_prob" in loss_dict:
            raw_metrics["gt_residual_log_prob"] = loss_dict["gt_residual_log_prob"]
        loss_dict.pop("gt_residual_log_prob", None)
        metrics = {
            k: (v.float().mean() if isinstance(v, torch.Tensor) else np.asarray(v).mean())
            for k, v in metrics.items()
        }
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
        if f"{prefix}spread_invisible_kp3d_samples" in metrics:
            self.log(
                "sample_kp3d_diversity_invis",
                metrics[f"{prefix}spread_invisible_kp3d_samples"],
                prog_bar=(prefix == "train_"),
                logger=False,
            )

        self.log(f"{prefix}loss", loss_dict[f"{prefix}total_loss"], prog_bar=True)
        self.log_dict(metrics, sync_dist=True)
        self.log_dict(loss_dict, sync_dist=True)

        if getattr(self.trainer, "sanity_checking", False):
            return None

        if prefix == "train_":
            vis_step = int(self.global_step)
            should_visualize = self.always_visualise or (
                vis_step in [2, 50, 250, 500, 1000, 2000, 3000, 4000]
                or (vis_step > 4000 and vis_step % 5000 == 0)
            )
        else:
            # For validation/test: one visualization per dataloader per epoch
            vis_step = int(
                self.global_step if self.global_step > 0 else (batch_idx or 0)
            )
            should_visualize = batch_idx == 0
        global_rank = getattr(self, "global_rank", 0)
        if should_visualize and global_rank == 0:
            # if global_rank == 0:
            image = batch["img_ori"][0]  # H W 3, bedlam 720 1280 3
            if isinstance(image, torch.Tensor):
                image = image.cpu().detach().numpy()  # [3, H, W]

            # Generate visualizations
            rend_img = my_visualize(
                image, outputs, self.faces, stack_vertically=self.stack_vertically, batch=batch
            )
            affine = batch["affine_trans"][0, 0]
            img_size = batch["img_size"][0, 0]
            rend_img_samples_crops = my_visualize_samples(
                image,
                outputs,
                self.faces,
                stack_vertically=False,  # self.stack_vertically,
                affine=affine,
                img_size=img_size,
                overlay_gt=True,
                plot_side=True,
                batch=batch,
                mhr_model=self.model.head_pose,
                metrics=raw_metrics,
            )
            rend_img_bgr = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
            rend_img_samples_crops_bgr = cv2.cvtColor(
                rend_img_samples_crops, cv2.COLOR_RGB2BGR
            )
            # Build filenames with unified format:
            # - Train: ep_xxx_train_xxxxxx_*.png
            # - Val:   ep_xxx_val[_dataset]_*.png
            epoch_part = f"ep_{self.current_epoch:03d}"
            if prefix == "train_":
                split_part = "train"
                step_part = f"{vis_step:06d}"
                base = f"{epoch_part}_{split_part}_{step_part}"
            else:
                split_part = "val"
                dataset_name = batch.get("dataset_name", ["unknown"])[0]
                if hasattr(dataset_name, "item"):
                    dataset_name = dataset_name.item()
                dataset_name = str(dataset_name)
                base = f"{epoch_part}_{split_part}_{dataset_name}"

            img_name = f"{base}_img.png"
            samples_name = f"{base}_samples_crops.png"

            cv2.imwrite(os.path.join(self.vis_save_dir, img_name), rend_img_bgr)
            cv2.imwrite(
                os.path.join(self.vis_save_dir, samples_name),
                rend_img_samples_crops_bgr,
            )

            # Build split name for Visualiser so filenames include epoch & dataset
            if prefix == "train_":
                split = "train"
            else:
                dataset_name = batch.get("dataset_name", ["unknown"])[0]
                if hasattr(dataset_name, "item"):
                    dataset_name = dataset_name.item()
                split = f"val_{str(dataset_name)}"

            self.visualiser.visualise(
                outputs,
                batch,
                batch_idx=batch_idx,
                split=split,
                epoch=self.current_epoch,
                global_step=vis_step,
            )
        return None

    def forward(self, batch: Dict, num_samples: int = 0) -> Dict:
        return self.model(batch, num_samples)

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0):
        batch = self.preprocess(batch)
        outputs = self(batch, num_samples=self.cfg.MODEL.NUM_SAMPLES)
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
        outputs = self(batch, num_samples=self.cfg.MODEL.NUM_SAMPLES)
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
            gt_joint_coords = gt_joint_coords @ R.transpose(-2, -1)

        batch["gt_verts_w_transl"] = gt_verts
        batch["gt_joint_coords"] = gt_joint_coords

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

        if "visibility" not in batch: # eg. 4d-dress 
            batch["visibility"] = torch.ones_like(batch["keypoints_2d"][:, :70, 0]).bool()

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
        
        if batch["dataset_name"][0] == "4d-dress":
            R = batch["cam_ext"][:, :3, :3]
            global_rotmat = global_rotmat @ R.transpose(-2, -1)

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
            shuffle=True,
            drop_last=True,
            collate_fn=bedlam_collate,
        )

    def val_dataset(self):
        datasets = self.cfg.DATASET.VAL_DS.split("_")
        logger.info(f"Validation datasets are: {datasets}")
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                BEDLAMDataset(
                    options=self.cfg.DATASET,
                    dataset=dataset_name,
                )
            )
        from sam_3d_body.data.d4dress_dataset import D4DressDataset
        val_datasets.append(D4DressDataset(cfg=self.cfg, ids=None))
        return val_datasets

    def val_dataloader(self):
        self.val_ds = self.val_dataset()
        dataloaders = []
        for val_ds in self.val_ds:
            loader_kw = dict(
                dataset=val_ds,
                batch_size=self.cfg.DATASET.BATCH_SIZE,
                shuffle=False,
                num_workers=self.cfg.DATASET.NUM_WORKERS,
                drop_last=False,
            )
            if isinstance(val_ds, BEDLAMDataset):
                loader_kw["collate_fn"] = bedlam_collate
            dataloaders.append(DataLoader(**loader_kw))
        return dataloaders
        # return DataLoader(
        #     dataset=self.val_ds,
        #     batch_size=self.cfg.DATASET.BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=self.cfg.DATASET.NUM_WORKERS,
        #     pin_memory=self.cfg.DATASET.PIN_MEMORY,
        #     drop_last=False,
        # )

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
                "/scratches/kyuban/cq244/datasets/SSP-3D/ssp_3d",
                num_view=num_view,
                cfg=self.cfg,
            )
        elif self.cfg.DATASET.VAL_DS == "4d-dress":
            from sam_3d_body.data.d4dress_dataset import MultiD4DressDataset

            logger.info(f"4D-DRESS dataset with num_view={num_view}")

            return MultiD4DressDataset(ids=None, cfg=self.cfg, num_views=num_view)

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

    @torch.no_grad()
    def run_multiview_prediction(
        self,
        num_view: int = 4,
        num_samples: int = 100,
        max_batches: Optional[int] = 2,
        dataset_name: str = "4d-dress",
        merge_method: str = "psis",
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
        dataloader = self.multiview_eval_dataloader(
            num_view=num_view, 
            batch_size=1, 
            dataset_name=dataset_name,
        )

        self.model.eval()

        all_metrics = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            bs, num_views = batch["img"].shape[:2]
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_views:
                        batch[k] = v.flatten(0, 1)

            batch = self.preprocess(batch)

            outputs = self.model(
                batch, 
                num_samples=num_samples
            )
            mhr_out = outputs["mhr"]
            uncertainty_out = outputs["uncertainty_output"]

            param_dict = merge_params_nf(
                self.model.nf_head,
                mhr_out,
                uncertainty_out,
                bs,
                num_views,
                num_samples,
                method=merge_method,
            )

            outs = get_mhr_outputs(
                mhr_head=self.model.head_pose,
                mhr_out=mhr_out,
                param_dict=param_dict,
                batch=batch,
                bs=bs,
                num_views=num_views,
                uncertainty_out=uncertainty_out,
            )

            all_metrics = multiframe_metrics(
                all_metrics, 
                outs,
                batch_idx=batch_idx,
                save_dir=self.vis_save_dir
            )

            renderer = Renderer(
                focal_length=outputs["mhr"]["focal_length"][0], 
                faces=self.faces
            )
            neutral_renderer = Renderer(
                focal_length=512, 
                faces=self.faces
            )

            outs.update(
                {
                    "renderer": renderer,
                    "neutral_renderer": neutral_renderer,
                    "outputs": outputs,
                    "batch": batch,
                    "metrics": all_metrics,
                    "num_views": num_views,
                    "bs": bs,
                    "batch_idx": batch_idx,
                }
            )

            
            # export_meshes_for_blender(outs, self.faces, self.vis_save_dir, tag=f"b{batch_idx:03d}")

            # if batch_idx == 3:
            #     for k, v in outs['metrics'].items():
            #         print(k, v)
            #     import ipdb; ipdb.set_trace()

            # vis_predictions(outs, sc=True, save_dir=self.vis_save_dir)
            # vis_neutral(outs, sc=True, save_dir=self.vis_save_dir, use_best_by_log_prob=True)

            # vis_predictions(outs, sc=False, save_dir=self.vis_save_dir)
            # vis_neutral(outs, sc=False, save_dir=self.vis_save_dir, plot_hist=True)

            # # ---------------------- Cross-view shape visualization ----------------------
            # # Get affine and img_size for cropping (first batch element = first num_views entries)
            # affine_all = batch["affine_trans"][:num_views]  # [num_views, 2, 3]
            # img_size_all = batch["img_size"][:num_views]  # [num_views, 2]

            # _, cross_view_gallery = view_one_in_another(
            #     outputs=outputs,
            #     batch=batch,
            #     mhr_model=self.model.head_pose,
            #     faces=self.faces,
            #     num_views=num_views,
            #     batch_idx=0,  # Visualize first batch element
            #     affine=affine_all,
            #     img_size=img_size_all,
            # )
            # cross_view_gallery_bgr = cv2.cvtColor(cross_view_gallery, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(
            #     os.path.join(self.vis_save_dir, f"b{batch_idx:03d}_cross_view.png"),
            #     cross_view_gallery_bgr,
            # )


        mean_metrics = print_multiview_metrics(all_metrics, self.vis_save_dir)

        return None
