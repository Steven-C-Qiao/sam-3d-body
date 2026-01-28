import os
import cv2
import numpy as np
import torch 
from typing import Dict, Optional
import roma

from yacs.config import CfgNode
import pytorch_lightning as pl
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



class Trainer(BaseLightningModule):
    """
    Trainer class that extends SAM3DBody with PyTorch Lightning training logic.
    Inherits all model functionality from SAM3DBody.
    """
    def __init__(self, cfg: CfgNode, vis_save_dir: str = None):
        super().__init__()

        self.cfg = cfg
        self.vis_save_dir = vis_save_dir

        # Select model based on config
        self.model_type = cfg.TRAIN.get("MODEL_TYPE", "full")
        if self.model_type == "toy":
            assert False 
            self.model = ToyModel(cfg)
        elif self.model_type == "full":
            self.model = SAM3DBody(cfg)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {self.model_type}. Must be 'toy' or 'full'.")

        self.metrics = Metrics()

        self.train_ds = self.train_dataset()
        self.val_ds = self.val_dataset()

        # Optionally enable dense keypoints based on config; if disabled, the model
        # will only use the canonical 70 MHR keypoints.
        self.use_dense_keypoints = bool(getattr(self.cfg.MODEL, "DENSE_KEYPOINTS", False))
        self.mhr_dense_kp_indices = None
        if self.use_dense_keypoints:
            mhr_dense_kp_indices_np = np.load(INDICES_PATH)
            self.mhr_dense_kp_indices = torch.from_numpy(mhr_dense_kp_indices_np).long()
            # Expose to the meta-arch and the MHR head for dense keypoint extraction
            setattr(self.model, "mhr_dense_kp_indices", self.mhr_dense_kp_indices)
            setattr(self.model.head_pose, "mhr_dense_kp_indices", self.mhr_dense_kp_indices)


        # Load checkpoint only for full model (toy model doesn't have pretrained weights)
        if self.model_type == "full":
            checkpoint = torch.load(cfg.TRAIN.CKPT_PATH, map_location='cpu', weights_only=False)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head_pose.shape_uncertainty_proj.parameters():
                param.requires_grad = True
            for param in self.model.head_pose.scale_uncertainty_proj.parameters():
                param.requires_grad = True


        self.scale_mean = self.model.head_pose.scale_mean.float()
        self.scale_comps = self.model.head_pose.scale_comps.float()
        
        
        self.criterion = Loss(cfg, scale_mean=self.scale_mean, scale_comps=self.scale_comps)

        self.faces = self.model.head_pose.faces.cpu().detach().numpy()
        
        self.visualiser = Visualiser(vis_save_dir, cfg=cfg, faces=self.faces)
        

    def training_step(self, batch: Dict, batch_idx: int):
        batch = self.preprocess(batch)
        
        outputs = self(batch, num_samples=5)

        loss_dict = self.criterion(outputs, batch)

        metrics = self.metrics(outputs, batch)


        self.log_metrics(loss_dict, metrics, batch, outputs)

        return loss_dict['total_loss']

    def log_metrics(self, loss_dict: Dict, metrics: Dict, batch: Dict, outputs: Dict):
        prog_bar_keys = ['kp2d_l1', 'kp2d_l1_samples', 'pampjpe', 'pampjpe_samples']
        prog_bar_metrics = {key: metrics.pop(key) for key in prog_bar_keys if key in metrics}

        if prog_bar_metrics:
            self.log_dict(prog_bar_metrics, on_step=True, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        
        if metrics:
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, rank_zero_only=True, sync_dist=True)

        # Log images every 500 steps to reduce overhead
        if self.global_step % 500 == 0:
        # if True:
            image = batch['img_ori'][0].data # H W 3, bedlam 720 1280 3 
            # image = batch['img'][0,0].data # [3, 256, 256] - CHW format, normalized
            image = image.cpu().detach().numpy() # [3, H, W]
            
            # Generate visualizations
            rend_img = my_visualize(image, outputs, self.faces)
            rend_img_samples = my_visualize_samples(image, outputs, self.faces)

            rend_img_bgr = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
            rend_img_samples_bgr = cv2.cvtColor(rend_img_samples, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.vis_save_dir, f'{self.global_step:06d}_img.png'), rend_img_bgr)
            cv2.imwrite(os.path.join(self.vis_save_dir, f'{self.global_step:06d}_samples.png'), rend_img_samples_bgr)

            self.visualiser.visualise(outputs, batch, batch_idx=None, global_step=self.global_step)

        self.log('train_loss', loss_dict['total_loss'], prog_bar=True)
        # self.model._log_metric('train_loss', loss_dict['total_loss'], step=batch_idx)
        self.log_dict(loss_dict)

        # import ipdb; ipdb.set_trace()

        return loss_dict['total_loss']


    def forward(self, batch: Dict, num_samples: int = 0) -> Dict:
        return self.model(batch, num_samples)
    


    def validation_step(self, batch: Dict, batch_idx: int):
        batch = self.preprocess(batch)

        outputs = self(batch, num_samples=5)

        loss_dict = self.criterion(outputs, batch)
        loss = loss_dict['total_loss']

        self.model._log_metric('val_loss', loss, step=batch_idx)
        
        metrics = self.metrics(outputs, batch)
        
        prog_bar_keys = ['kp2d_l1', 'kp2d_l1_samples', 'pampjpe', 'pampjpe_samples']
        prog_bar_metrics = {key: metrics.pop(key) for key in prog_bar_keys if key in metrics}
        
        if prog_bar_metrics:
            self.log_dict(prog_bar_metrics, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        if metrics:
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=True, sync_dist=True)

        return loss
    


    def preprocess(self, batch: Dict):
        mhr_model = self.model.head_pose 
        gt_mhr_output = mhr_model.mhr(
            identity_coeffs=batch['shape_params'],
            model_parameters=batch['model_params'],
            face_expr_coeffs=batch['face_expr_coeffs'],
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
            (
                mhr_model.keypoint_mapping
                @ gt_vert_joints.permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, gt_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )

        cam_int = batch['cam_int']
        cam_ext = batch['cam_ext'] 
        trans_cam = cam_ext[:, :3, 3]

        def project(points, cam_trans, cam_int):
            points = points + cam_trans
            # Normalize by Z (divide by last coordinate)
            projected_points = points / points[..., -1].unsqueeze(-1)
            # Multiply by camera intrinsics: cam_int @ projected_points.T
            projected_points = torch.einsum('bij, bkj->bki', cam_int, projected_points)
            return projected_points
        
        keypoints_2d_by_projection = project(gt_keypoints_3d, trans_cam.unsqueeze(1), cam_int)[:, :70, :2]

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

        

        gt_kp2d_h = torch.cat(
            [kp2d, torch.ones_like(kp2d[..., :1])], dim=-1
        ).float()
        affine = batch["affine_trans"][:, 0].float()
        img_size = batch["img_size"][:, 0]

        gt_kp2d_crop = gt_kp2d_h @ affine.mT  # [B, 70, 3] @ [B, 3, 2] = [B, 70, 2]
        gt_kp2d_crop = gt_kp2d_crop[..., :2]

        # Normalize to [-0.5, 0.5], same as _full_to_crop
        # img_size needs to be [B, 1, 2] for proper broadcasting
        # gt_kp2d_crop = gt_kp2d_crop / img_size.unsqueeze(1) - 0.5  # [B, 70, 2]

        batch['keypoints_2d'] = gt_kp2d_crop
        

        model_parameters = batch['model_params']
        model_parameters[:, :3] = 0
        global_rot = batch['model_params'][:, 3:6]
        
        # Add 180-degree rotation around X-axis
        global_rot_mat = roma.euler_to_rotmat("xyz", global_rot)  # B x 3 x 3
        
        batch_size = global_rot.shape[0]
        rot_180_x = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=global_rot.dtype, device=global_rot.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multiply rotations: new_rot = rot_180_x @ global_rot_mat
        new_global_rot_mat = torch.bmm(rot_180_x, global_rot_mat)
        
        # Convert back to Euler angles
        global_rot = roma.rotmat_to_euler("xyz", new_global_rot_mat)
        
        # Update model_parameters with the new global_rot
        model_parameters[:, 3:6] = global_rot
        
        gt_mhr_output = mhr_model.mhr(
            identity_coeffs=batch['shape_params'],
            model_parameters=model_parameters,
            face_expr_coeffs=batch['face_expr_coeffs'],
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
            (
                mhr_model.keypoint_mapping
                @ gt_vert_joints.permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, gt_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )

        # Ground-truth 3D keypoints: always include the canonical 70 MHR keypoints,
        # and optionally append dense keypoints if enabled.
        gt_kp3d_70 = gt_keypoints_3d_all[:, :70]  # [B, 70, 3]
        if self.use_dense_keypoints and self.mhr_dense_kp_indices is not None:
            dense_kp3d_gt = gt_verts[:, self.mhr_dense_kp_indices, :]  # [B, N_dense, 3]
            gt_keypoints_3d = torch.cat([gt_kp3d_70, dense_kp3d_gt], dim=1)  # [B, 70+N_dense, 3]
        else:
            gt_keypoints_3d = gt_kp3d_70

        batch['joints_3d'] = gt_joint_coords
        batch['vertices'] = gt_verts
        batch['keypoints_3d'] = gt_keypoints_3d

        return batch 


    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.TRAIN.LR)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """
        Override to only save trainable (unfrozen) parameters in checkpoints.
        This reduces checkpoint size significantly when most parameters are frozen.
        """
        # Get the full state dict
        state_dict = checkpoint['state_dict']
        
        # Get all trainable parameter names from the model
        trainable_param_names = set()
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                trainable_param_names.add(name)
                trainable_params += 1
        
        # Filter state_dict to only include trainable parameters
        # PyTorch Lightning adds 'model.' prefix to model parameters
        filtered_state_dict = {}
        saved_model_params = 0
        
        # Keys to exclude: scale_mean, scale_comps, scale_comps_pinv (extracted from loaded model, not trained)
        exclude_keys = {'scale_mean', 'scale_comps', 'scale_comps_pinv'}
        
        for key, value in state_dict.items():
            # Skip scale-related keys that shouldn't be saved
            if any(excluded in key for excluded in exclude_keys):
                continue
            
            # Check if this is a model parameter
            if key.startswith('model.'):
                # Remove 'model.' prefix to get the actual parameter name
                param_name = key[6:]  # Remove 'model.' prefix
                # Only include if it's a trainable parameter
                if param_name in trainable_param_names:
                    filtered_state_dict[key] = value
                    saved_model_params += 1
            else:
                # Keep non-model keys (optimizer states, etc.)
                filtered_state_dict[key] = value
        
        # Log checkpoint saving info
        logger.info(f"Saving checkpoint: {saved_model_params}/{total_params} trainable parameters "
                   f"({100*saved_model_params/total_params:.1f}% of model parameters)")
        
        # Update checkpoint with filtered state dict
        checkpoint['state_dict'] = filtered_state_dict
        
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """
        Override to handle loading checkpoints that only contain trainable parameters.
        Frozen parameters will keep their original loaded values.
        """
        # The checkpoint only contains trainable parameters
        # Load them with strict=False so frozen parameters are not affected
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Filter to only model parameters (remove optimizer states, etc.)
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    # Remove 'model.' prefix for loading
                    param_name = key[6:]
                    model_state_dict[param_name] = value
            
            if model_state_dict:
                self.model.load_state_dict(model_state_dict, strict=False)




    def train_dataset(self):
        options = self.cfg.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
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
            shuffle=self.cfg.DATASET.SHUFFLE_TRAIN,
            drop_last=True
        )

    def val_dataset(self):
        datasets = self.cfg.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                BEDLAMDataset(
                    options=self.cfg.DATASET,
                    dataset=dataset_name,
                    is_train=False,
                )
            )
        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.cfg.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.cfg.DATASET.NUM_WORKERS,
                    drop_last=True
                )
            )
        return dataloaders
    

    def multiview_eval_dataset(self, num_view: int = 4):
        """
        Build a BEDLAM multi-view evaluation dataset using MultiViewEvaluationDataset.
        
        Each sample corresponds to a unique serial number (serno) and contains
        `num_view` different camera views of the same subject.
        """
        options = self.cfg.DATASET
        # Use the same datasets as training by default (first dataset only)
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_name = dataset_names[0]

        logger.info(
            f"Creating MultiViewEvaluationDataset for '{dataset_name}' "
            f"with num_view={num_view}"
        )

        multiview_ds = MultiViewEvaluationDataset(
            options=options,
            dataset=dataset_name,
            num_view=num_view,
            is_train=True,  # uses training BEDLAM splits
        )

        return multiview_ds

    def multiview_eval_dataloader(self, num_view: int = 4, batch_size: int = 1):
        """
        DataLoader wrapping the multi-view evaluation dataset.
        
        Batch size defaults to 1 so that each batch corresponds to a single serno,
        with `num_view` views.
        """
        multiview_ds = self.multiview_eval_dataset(num_view=num_view)
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
        device = next(self.model.parameters()).device
        print(f"Device: {device}")
        
        dataloader = self.multiview_eval_dataloader(num_view=num_view, batch_size=1)

        all_results = []

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            

            # Move tensor fields to device
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Batch size is 1 by construction
            # Shapes: [1, V, ...] -> we work over the view dimension
            num_views = int(batch["num_views"][0].item() if isinstance(batch["num_views"], torch.Tensor) else batch["num_views"][0])

            # Some fields may be simple Python lists (e.g. serno, selected_indices)
            serno = batch.get("selected_serno", [None])[0]
            indices = batch.get("selected_indices", [None])[0]

            # Track shape and scale parameters (mu) and their uncertainties (sigma) for each view
            shape_params_per_view = []
            scale_params_per_view = []
            shape_uncertainties_per_view = []
            scale_uncertainties_per_view = []
            
            # Also track per-view predictions for reference
            pred_vertices_per_view = []
            pred_joints_per_view = []
            # Track ground truth vertices per view
            gt_vertices_per_view = []
            # Track GT shape and scale parameters per view (for neutral pose generation)
            gt_shape_per_view = []
            gt_scale_per_view = []
            # Track per-view pose parameters for later use with merged shape/scale
            global_rot_per_view = []
            body_pose_per_view = []
            hand_pose_per_view = []
            face_expr_per_view = []

            for view_idx in range(num_views):
                # Create a single-view batch for this view
                # The model expects [B, N, ...] where N is number of persons (not views)
                view_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # Extract this view's data and eliminate view dimension
                        if value.dim() > 1 and value.shape[1] == num_views:
                            # [B, V, ...] -> [B, ...] by selecting view_idx
                            view_batch[key] = value[:, view_idx]
                        else:
                            view_batch[key] = value
                    elif isinstance(value, list):
                        # For list fields, just take this view's item
                        view_batch[key] = [value[view_idx]] if view_idx < len(value) else value
                    else:
                        view_batch[key] = value

                # Preprocess view_batch to get GT vertices (similar to training/validation step)
                view_batch = self.preprocess(view_batch)
                
                # Run full model forward pass to get predictions and uncertainties
                with torch.no_grad():
                    outputs = self.model(view_batch, num_samples=0)
                
                # Extract predicted shape and scale parameters (mu)
                pred_shape = outputs['mhr']['shape'][0]  # [num_shape_comps]
                pred_scale = outputs['mhr']['scale'][0]  # [num_scale_comps]
                
                # Extract uncertainties (these are variances, shape: [1, D])
                shape_uncertainty = outputs['mhr']['shape_uncertainty'][0]  # [num_shape_comps]
                scale_uncertainty = outputs['mhr']['scale_uncertainty'][0]  # [num_scale_comps]
                
                # Also extract vertices/joints for reference
                pred_verts = outputs['mhr']['pred_vertices'][0]  # [N_verts, 3]
                pred_joints = outputs['mhr']['pred_keypoints_3d'][0]  # [N_joints, 3]
                
                # Extract ground truth vertices for this view
                # The batch should have vertices after preprocessing
                gt_verts = view_batch['vertices']
                if isinstance(gt_verts, torch.Tensor):
                    if gt_verts.dim() > 1:
                        gt_verts = gt_verts[0]  # Take first person if [B, N, 3]
                    gt_vertices_per_view.append(gt_verts.detach().cpu())
                else:
                    gt_vertices_per_view.append(gt_verts)
                
                # Extract GT shape and scale parameters for neutral pose generation
                # GT shape is directly available
                gt_shape = view_batch['shape_params']
                if isinstance(gt_shape, torch.Tensor):
                    if gt_shape.dim() > 1:
                        gt_shape = gt_shape[0]  # Take first person if [B, N, ...]
                    gt_shape_per_view.append(gt_shape.detach().cpu())
                else:
                    gt_shape_per_view.append(gt_shape)
                
                # GT scale: extract actual scales (68 dims) directly, avoid unstable projection to lower dimensions
                device = next(self.model.parameters()).device
                gt_model_params = view_batch['model_params']
                if isinstance(gt_model_params, torch.Tensor):
                    if gt_model_params.dim() > 1:
                        gt_model_params = gt_model_params[0]  # Take first person if [B, N, ...]
                    # Extract actual scales (last 68 elements)
                    gt_actual_scales = gt_model_params[-68:].to(device)  # [68]
                    gt_scale_per_view.append(gt_actual_scales.detach().cpu())
                else:
                    # If not tensor, try to convert
                    gt_actual_scales = torch.tensor(gt_model_params[-68:], device=device)
                    gt_scale_per_view.append(gt_actual_scales.detach().cpu())
                
                # Extract pose parameters for this view (to use with merged shape/scale later)
                global_rot = outputs['mhr']['global_rot'][0]  # [3]
                body_pose = outputs['mhr']['body_pose'][0]  # [body_pose_dim]
                hand_pose = outputs['mhr']['hand'][0]  # [hand_pose_dim]
                face_expr = outputs['mhr']['face'][0]  # [face_expr_dim]
                
                shape_params_per_view.append(pred_shape)
                scale_params_per_view.append(pred_scale)
                shape_uncertainties_per_view.append(shape_uncertainty)
                scale_uncertainties_per_view.append(scale_uncertainty)
                pred_vertices_per_view.append(pred_verts)
                pred_joints_per_view.append(pred_joints)
                global_rot_per_view.append(global_rot)
                body_pose_per_view.append(body_pose)
                hand_pose_per_view.append(hand_pose)
                face_expr_per_view.append(face_expr)

            # Stack shape and scale parameters (mu) and uncertainties (sigma)
            # mu_shape: [N_views, D_shape], mu_scale: [N_views, D_scale]
            mu_shape = torch.stack(shape_params_per_view, dim=0)  # [N_views, D_shape]
            mu_scale = torch.stack(scale_params_per_view, dim=0)  # [N_views, D_scale]
            
            # Stack uncertainties: [N_views, D_shape] and [N_views, num_selected_scales]
            # Note: scale_uncertainty is only predicted for selected indices
            shape_uncertainties = torch.stack(shape_uncertainties_per_view, dim=0)  # [N_views, D_shape]
            scale_uncertainties = torch.stack(scale_uncertainties_per_view, dim=0)  # [N_views, num_selected_scales]
            
            # Selected scale component indices (same as in forward pass)
            selected_scale_comps_indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
            num_selected_scales = len(selected_scale_comps_indices)
            
            # Construct diagonal covariance matrices from uncertainties
            # sigma_shape: [N_views, D_shape, D_shape]
            N_views = mu_shape.shape[0]
            D_shape = mu_shape.shape[1]
            D_scale = mu_scale.shape[1]
            
            sigma_shape_list = []
            sigma_scale_selected_list = []
            
            for view_idx in range(N_views):
                # Create diagonal covariance matrices from uncertainties (variances)
                sigma_shape = torch.diag(shape_uncertainties[view_idx])  # [D_shape, D_shape]
                # scale_uncertainty is only for selected indices
                sigma_scale_selected = torch.diag(scale_uncertainties[view_idx])  # [num_selected_scales, num_selected_scales]
                
                sigma_shape_list.append(sigma_shape)
                sigma_scale_selected_list.append(sigma_scale_selected)
            
            sigma_shape = torch.stack(sigma_shape_list, dim=0)  # [N_views, D_shape, D_shape]
            sigma_scale_selected = torch.stack(sigma_scale_selected_list, dim=0)  # [N_views, num_selected_scales, num_selected_scales]

            # Merge shape parameters (full)
            mu_star_shape, sigma_star_shape = self.merge_predictions(mu_shape, sigma_shape)

            # Merge scale parameters (only selected indices)
            # Extract selected scale components from mu_scale
            mu_scale_selected = mu_scale[:, selected_scale_comps_indices]  # [N_views, num_selected_scales]
            mu_star_scale_selected, sigma_star_scale_selected = self.merge_predictions(mu_scale_selected, sigma_scale_selected)
            
            # Reconstruct full scale parameters: merged values for selected indices, average for others
            mu_star_scale = mu_scale.mean(dim=0)  # [D_scale] - average of all views for non-selected
            mu_star_scale[selected_scale_comps_indices] = mu_star_scale_selected  # Replace with merged values
            
            # For sigma_star_scale, create a full covariance matrix
            # Since sigma_star_scale_selected is diagonal (from merging diagonal matrices),
            # we only need to fill in the diagonal elements
            default_uncertainty = scale_uncertainties.mean().item()
            sigma_star_scale = torch.eye(D_scale, device=mu_star_scale.device, dtype=mu_star_scale.dtype) * default_uncertainty
            # Fill in merged diagonal covariance for selected indices
            for i, idx in enumerate(selected_scale_comps_indices):
                sigma_star_scale[idx, idx] = sigma_star_scale_selected[i, i]
            
            # Create full sigma_scale for storing per-view uncertainties (for reference)
            # For selected indices, use actual uncertainties; for others, use default
            sigma_scale_list_full = []
            for view_idx in range(N_views):
                sigma_scale_full = torch.eye(D_scale, device=mu_scale.device, dtype=mu_scale.dtype) * default_uncertainty
                # Fill in selected indices with actual uncertainties
                for i, idx in enumerate(selected_scale_comps_indices):
                    sigma_scale_full[idx, idx] = scale_uncertainties[view_idx, i]
                sigma_scale_list_full.append(sigma_scale_full)
            sigma_scale = torch.stack(sigma_scale_list_full, dim=0)  # [N_views, D_scale, D_scale]

            # Run MHR forward with merged shape and scale parameters for EACH view
            # Use view-specific pose parameters but merged shape/scale
            device = next(self.model.parameters()).device
            mu_star_vertices_per_view = []
            mu_star_joints_per_view = []
            
            for view_idx in range(N_views):
                # Get view-specific pose parameters
                global_rot = global_rot_per_view[view_idx]  # [3]
                body_pose = body_pose_per_view[view_idx]  # [body_pose_dim]
                hand_pose = hand_pose_per_view[view_idx]  # [hand_pose_dim]
                face_expr = face_expr_per_view[view_idx]  # [face_expr_dim]
                
                # Run MHR with merged shape/scale but view-specific pose
                mhr_output = self.model.head_pose.mhr_forward(
                    global_trans=torch.zeros_like(outputs['mhr']['global_rot']),
                    global_rot=global_rot.unsqueeze(0),
                    body_pose_params=body_pose.unsqueeze(0),
                    hand_pose_params=hand_pose.unsqueeze(0),
                    scale_params=mu_star_scale.unsqueeze(0),
                    shape_params=mu_star_shape.unsqueeze(0),
                    expr_params=face_expr.unsqueeze(0),
                    return_keypoints=True,
                    return_joint_coords=True,
                    do_pcblend=True,
                )
                
                # mhr_forward returns (verts, j3d, jcoords, ...) when return_keypoints=True and return_joint_coords=True
                verts_star = mhr_output[0]  # [1, N_verts, 3]
                j3d_star = mhr_output[1]  # [1, N_joints, 3]
                jcoords_star = mhr_output[2]  # [1, N_joints, 3]
                
                # Apply camera system difference (same as in forward pass)
                verts_star[..., [1, 2]] *= -1
                j3d_star[..., [1, 2]] *= -1
                
                mu_star_vertices_per_view.append(verts_star[0].detach().cpu())  # [N_verts, 3]
                mu_star_joints_per_view.append(j3d_star[0, :70].detach().cpu())  # [70, 3] - first 70 keypoints

            # Stack lists into tensors
            pred_vertices_stacked = torch.stack(pred_vertices_per_view, dim=0)  # [N_views, N_verts, 3]
            pred_joints_stacked = torch.stack(pred_joints_per_view, dim=0)  # [N_views, N_joints, 3]
            mu_star_vertices_stacked = torch.stack(mu_star_vertices_per_view, dim=0)  # [N_views, N_verts, 3]
            mu_star_joints_stacked = torch.stack(mu_star_joints_per_view, dim=0)  # [N_views, 70, 3]
            
            # Stack GT vertices
            gt_vertices_stacked = torch.stack(gt_vertices_per_view, dim=0)  # [N_views, N_verts, 3]
            
            # Generate neutral pose bodies (zero pose, keep shape/scale)
            # For GT: use GT shape/scale per view
            # For predicted: use predicted shape/scale per view
            # For merged: use merged shape/scale (same for all views)
            device = next(self.model.parameters()).device
            gt_vertices_neutral_per_view = []
            pred_vertices_neutral_per_view = []
            merged_vertices_neutral_per_view = []
            
            # Zero pose parameters for neutral pose
            zero_global_trans = torch.zeros(3, device=device)
            zero_global_rot = torch.zeros(3, device=device)
            zero_body_pose = torch.zeros(130, device=device)  # body_pose_dim = 130
            zero_hand_pose = torch.zeros(108, device=device)  # hand_pose_dim = 54 * 2
            zero_face_expr = torch.zeros(72, device=device)  # face_expr_dim = 72
            
            for view_idx in range(N_views):
                # GT neutral pose
                # Use actual scales directly via scale_offsets to avoid unstable projection
                gt_shape_view = gt_shape_per_view[view_idx].to(device)
                gt_actual_scales_view = gt_scale_per_view[view_idx].to(device)  # [68] actual scales
                # Use scale_offsets to provide actual scales directly: scales = scale_mean + 0*scale_comps + (actual_scales - scale_mean)
                gt_scale_offsets = gt_actual_scales_view - self.scale_mean.to(device)  # [68]
                zero_scale_params = torch.zeros(28, device=device)  # [28] zero scale params
                gt_neutral_output = self.model.head_pose.mhr_forward(
                    global_trans=zero_global_trans.unsqueeze(0),
                    global_rot=zero_global_rot.unsqueeze(0),
                    body_pose_params=zero_body_pose.unsqueeze(0),
                    hand_pose_params=zero_hand_pose.unsqueeze(0),
                    scale_params=zero_scale_params.unsqueeze(0),
                    shape_params=gt_shape_view.unsqueeze(0),
                    expr_params=zero_face_expr.unsqueeze(0),
                    scale_offsets=gt_scale_offsets.unsqueeze(0),
                    return_keypoints=True,
                    return_joint_coords=True,
                    do_pcblend=True,
                )
                gt_neutral_verts = gt_neutral_output[0][0]  # [N_verts, 3]
                # gt_neutral_verts[..., [1, 2]] *= -1  # Camera system difference
                gt_vertices_neutral_per_view.append(gt_neutral_verts.detach().cpu())
                
                # Predicted neutral pose (using predicted shape/scale for this view)
                pred_shape_view = shape_params_per_view[view_idx].to(device)
                pred_scale_view = scale_params_per_view[view_idx].to(device)
                pred_neutral_output = self.model.head_pose.mhr_forward(
                    global_trans=zero_global_trans.unsqueeze(0),
                    global_rot=zero_global_rot.unsqueeze(0),
                    body_pose_params=zero_body_pose.unsqueeze(0),
                    hand_pose_params=zero_hand_pose.unsqueeze(0),
                    scale_params=pred_scale_view.unsqueeze(0),
                    shape_params=pred_shape_view.unsqueeze(0),
                    expr_params=zero_face_expr.unsqueeze(0),
                    return_keypoints=True,
                    return_joint_coords=True,
                    do_pcblend=True,
                )
                pred_neutral_verts = pred_neutral_output[0][0]  # [N_verts, 3]
                pred_neutral_verts[..., [1, 2]] *= -1  # Camera system difference
                pred_vertices_neutral_per_view.append(pred_neutral_verts.detach().cpu())
                
                # Merged neutral pose (using merged shape/scale, same for all views)
                merged_neutral_output = self.model.head_pose.mhr_forward(
                    global_trans=zero_global_trans.unsqueeze(0),
                    global_rot=zero_global_rot.unsqueeze(0),
                    body_pose_params=zero_body_pose.unsqueeze(0),
                    hand_pose_params=zero_hand_pose.unsqueeze(0),
                    scale_params=mu_star_scale.unsqueeze(0).to(device),
                    shape_params=mu_star_shape.unsqueeze(0).to(device),
                    expr_params=zero_face_expr.unsqueeze(0),
                    return_keypoints=True,
                    return_joint_coords=True,
                    do_pcblend=True,
                )
                merged_neutral_verts = merged_neutral_output[0][0]  # [N_verts, 3]
                merged_neutral_verts[..., [1, 2]] *= -1  # Camera system difference
                merged_vertices_neutral_per_view.append(merged_neutral_verts.detach().cpu())
            
            # Stack neutral pose vertices
            gt_vertices_neutral_stacked = torch.stack(gt_vertices_neutral_per_view, dim=0)  # [N_views, N_verts, 3]
            pred_vertices_neutral_stacked = torch.stack(pred_vertices_neutral_per_view, dim=0)  # [N_views, N_verts, 3]
            merged_vertices_neutral_stacked = torch.stack(merged_vertices_neutral_per_view, dim=0)  # [N_views, N_verts, 3]
            
            result = {
                "serno": serno,
                "indices": indices,
                # Ground truth vertices per view
                "gt_vertices": gt_vertices_stacked,  # [N_views, N_verts, 3] or None - GT vertices per view
                # Per-view predictions (for reference)
                "pred_vertices": pred_vertices_stacked,  # [N_views, N_verts, 3] - predicted vertices per view
                "pred_joints": pred_joints_stacked,  # [N_views, N_joints, 3] - predicted joints per view
                # Shape parameters: mu and sigma
                "mu_shape": mu_shape,  # [N_views, D_shape] - predicted shape params per view
                "sigma_shape": sigma_shape,  # [N_views, D_shape, D_shape] - shape uncertainties (covariances)
                "mu_star_shape": mu_star_shape,  # [D_shape] - merged shape params
                "sigma_star_shape": sigma_star_shape,  # [D_shape, D_shape] - merged shape uncertainty
                # Scale parameters: mu and sigma
                "mu_scale": mu_scale,  # [N_views, D_scale] - predicted scale params per view
                "sigma_scale": sigma_scale,  # [N_views, D_scale, D_scale] - scale uncertainties (covariances)
                "mu_star_scale": mu_star_scale,  # [D_scale] - merged scale params
                "sigma_star_scale": sigma_star_scale,  # [D_scale, D_scale] - merged scale uncertainty
                # Final merged predictions (vertices and joints from merged parameters, per view)
                "mu_star_vertices": mu_star_vertices_stacked,  # [N_views, N_verts, 3] - merged vertices per view
                "mu_star_joints": mu_star_joints_stacked,  # [N_views, 70, 3] - merged joints per view
                # Neutral pose vertices (zero pose, keep shape/scale)
                "gt_vertices_neutral": gt_vertices_neutral_stacked,  # [N_views, N_verts, 3] - GT neutral pose per view
                "pred_vertices_neutral": pred_vertices_neutral_stacked,  # [N_views, N_verts, 3] - predicted neutral pose per view
                "merged_vertices_neutral": merged_vertices_neutral_stacked,  # [N_views, N_verts, 3] - merged neutral pose per view
            }
            all_results.append(result)

            # Visualize posed bodies
            self.visualiser.visualise_merging(result, batch, suffix=None, normalise=False)
            
            # Visualize neutral pose bodies (with height normalization by default)
            neutral_result = {
                'gt_vertices': result['gt_vertices_neutral'],
                'pred_vertices': result['pred_vertices_neutral'],
                'mu_star_vertices': result['merged_vertices_neutral'],
            }
            self.visualiser.visualise_merging(neutral_result, batch, suffix='neutral', normalise=True)



        # Stack all results to restore batch dimension
        if len(all_results) == 0:
            return {}
        
        # Collect all values for each key
        stacked_results = {}
        for key in all_results[0].keys():
            values = [result[key] for result in all_results]
            
            # Handle tensor fields - stack them
            if isinstance(values[0], torch.Tensor):
                stacked_results[key] = torch.stack(values, dim=0)  # [B, ...]
            # Handle list fields - keep as list or convert to tensor if all are tensors
            elif isinstance(values[0], list):
                # Check if list contains tensors that can be stacked
                if len(values[0]) > 0 and isinstance(values[0][0], torch.Tensor):
                    # Stack each list element, then stack across batch
                    # This creates [B, N_views, ...] for per-view data
                    stacked_results[key] = torch.stack([torch.stack(v, dim=0) for v in values], dim=0)
                else:
                    # Keep as list of lists
                    stacked_results[key] = values
            else:
                # For non-tensor fields (e.g., serno, indices), keep as list
                stacked_results[key] = values
        
        return stacked_results
    
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
        
        mu_star = sigma_star @ torch.einsum('nij,nj->ni', inv_sigma, mu).sum(dim=0)

        return mu_star, sigma_star

    def visualize_keypoints_2d(
        self, 
        batch: Dict, 
        outputs: Dict, 
        batch_idx: int = 0,
        save_path: str = 'temp_vis.png'
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
        gt_kp2d = batch['keypoints_2d'][batch_idx, :, :].cpu().detach().numpy()  # [N, 2]
        pred_kp2d_cropped_normalised_coords = outputs['mhr']['pred_keypoints_2d_cropped'][batch_idx].cpu().detach().numpy()  # [70, 2]
        pred_kp2d_cropped_coords = (pred_kp2d_cropped_normalised_coords + 0.5) * 256  # [70, 2]
        
        # Get cropped image
        img = batch['img'][batch_idx, 0].clone().cpu().detach().numpy()  # C, H, W
        img = img.transpose(1, 2, 0)  # H, W, C
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(gt_kp2d[:, 0], gt_kp2d[:, 1], color='blue', s=10, marker='x', label='GT')
        plt.scatter(pred_kp2d_cropped_coords[:, 0], pred_kp2d_cropped_coords[:, 1], 
                   color='red', s=10, marker='o', label='Pred Cropped Coords')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_keypoints_2d_samples(
        self, 
        batch: Dict, 
        outputs: Dict, 
        batch_idx: int = 0,
        save_path: str = 'temp_vis_samples.png'
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
        
        if 'mhr_samples_keypoints_2d' not in outputs:
            logger.warning("No sample keypoints found in outputs. Skipping visualization.")
            return
        
        # Extract keypoints
        gt_kp2d = batch['keypoints_2d'][batch_idx, :, :].cpu().detach().numpy()  # [N, 2]
        pred_kp2d_cropped_normalised_coords = outputs['mhr']['pred_keypoints_2d_cropped'][batch_idx].cpu().detach().numpy()  # [70, 2]
        pred_kp2d_cropped_coords = (pred_kp2d_cropped_normalised_coords + 0.5) * 256  # [70, 2]
        
        # Extract sample keypoints (in full image coords)
        sample_kp2d_full = outputs['mhr_samples_keypoints_2d'][batch_idx].cpu().detach().numpy()  # [num_samples, 70, 2]
        
        # Convert samples to cropped coordinates
        num_samples = sample_kp2d_full.shape[0]
        sample_kp2d_cropped_coords = []
        
        # Ensure body_batch_idx is set (should be set during forward pass)
        if not hasattr(self.model, 'body_batch_idx') or len(self.model.body_batch_idx) == 0:
            self.model.body_batch_idx = [batch_idx]
        
        for i in range(num_samples):
            # Convert full image coords to cropped coords using _full_to_crop
            sample_kp2d_tensor = torch.from_numpy(sample_kp2d_full[i:i+1]).to(batch['img'].device)
            sample_kp2d_cropped_normalized = self.model._full_to_crop(
                batch, 
                sample_kp2d_tensor,
                torch.tensor([self.model.body_batch_idx[batch_idx]], device=batch['img'].device) if len(self.model.body_batch_idx) > batch_idx else None
            )[0].cpu().detach().numpy()  # [70, 2] in normalized cropped coords [-0.5, 0.5]
            
            # Convert to pixel coordinates
            sample_kp2d_cropped = (sample_kp2d_cropped_normalized + 0.5) * 256  # [70, 2]
            sample_kp2d_cropped_coords.append(sample_kp2d_cropped)
        
        sample_kp2d_cropped_coords = np.array(sample_kp2d_cropped_coords)  # [num_samples, 70, 2]
        
        # Get cropped image
        img = batch['img'][batch_idx, 0].clone().cpu().detach().numpy()  # C, H, W
        img = img.transpose(1, 2, 0)  # H, W, C
        
        # Plot samples visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(gt_kp2d[:, 0], gt_kp2d[:, 1], color='blue', s=15, marker='x', label='GT', linewidths=2)
        plt.scatter(pred_kp2d_cropped_coords[:, 0], pred_kp2d_cropped_coords[:, 1], 
                   color='red', s=15, marker='o', label='Pred Mean', linewidths=2, edgecolors='darkred')
        
        # Plot each sample with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
        for i in range(num_samples):
            plt.scatter(sample_kp2d_cropped_coords[i, :, 0], sample_kp2d_cropped_coords[i, :, 1],
                       color=colors[i], s=8, marker='.', alpha=0.6, 
                       label=f'Sample {i+1}' if i < 5 else None)  # Only label first 5 to avoid clutter
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
