# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional

import roma
import torch
import torch.nn as nn
import torchvision.models as models

from sam_3d_body.models.modules import rot6d_to_rotmat
from sam_3d_body.models.modules.mhr_utils import (
    compact_cont_to_model_params_body,
    mhr_param_hand_mask,
)
from sam_3d_body.models.modules.transformer import FFN

from .base_model import BaseModel


class ToyModel(BaseModel):
    """
    Simplified version of SAM3DBody using ResNet18 backbone with MLP heads.
    Predicts MHR pose, shape, and weak perspective camera parameters.
    """

    def _initialze_model(self):
        # Register image normalization buffers
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # ResNet18 backbone (remove final classification layer)
        resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # ResNet18 outputs 512-dim features after global average pooling
        
        # Feature dimension after ResNet18
        self.backbone_feat_dim = 512
        backbone_feat_dim = self.backbone_feat_dim
        
        # MHR pose head parameters
        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.body_cont_dim = 260
        
        # Total pose parameters: global_rot (6) + body_pose (260) + shape (45) + scale (28) + hand (54*2) + face (72)
        self.npose = (
            6  # Global rotation (6D)
            + self.body_cont_dim  # Body pose (continuous)
            + self.num_shape_comps  # Shape
            + self.num_scale_comps  # Scale
            + self.num_hand_comps * 2  # Hand (left + right)
            + self.num_face_comps  # Face
        )
        
        # MLP head for MHR pose prediction
        mlp_hidden_dim = backbone_feat_dim // 2
        self.head_pose_mlp = nn.Sequential(
            nn.Linear(backbone_feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, self.npose),
        )
        
        # Initialize the last layer bias to valid zero-pose (not all zeros!)
        # This ensures the model starts with a valid pose prediction
        from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body
        zero_pose_init = torch.zeros(1, self.npose)
        # Set global rotation to identity (6D representation: [1, 0, 0, 0, 1, 0])
        zero_pose_init[:, :6] = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        # Set body pose to zero-pose in continuous representation
        zero_pose_init[:, 6:6+self.body_cont_dim] = compact_model_params_to_cont_body(
            torch.zeros(1, 133)
        ).squeeze()
        # Shape, scale, hand, face remain zero (which is fine for initialization)
        self.head_pose_mlp[-1].bias.data = zero_pose_init.squeeze()
        
        # MLP head for camera parameters (s, tx, ty)
        self.head_camera = nn.Sequential(
            nn.Linear(backbone_feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, 3),  # (s, tx, ty)
        )
        nn.init.zeros_(self.head_camera[-1].bias)
        
        # Load MHR head for forward kinematics (reuse from existing head if available)
        # We'll need to get the MHR model from the config or create a minimal version
        # For now, we'll assume we can get it from build_head
        from sam_3d_body.models.heads import build_head
        
        # Create MHR uncertainty head for forward kinematics and uncertainty prediction
        # We use uncertainty head to get shape/scale uncertainty for sampling
        self.mhr_head = build_head(self.cfg, 'uncertainty')
        # Freeze MHR parameters (only uncertainty projections will be trainable if needed)
        for param in self.mhr_head.parameters():
            param.requires_grad = False
        
        # Add head_pose property for compatibility with Trainer code
        # This allows Trainer to access head_pose the same way for both models
        self.head_pose = self.mhr_head
        
        # Create camera head for projection (only used for projection, not prediction)
        from sam_3d_body.models.heads.camera_head import PerspectiveHead
        img_size = tuple(self.cfg.MODEL.IMAGE_SIZE)
        self.camera_head = PerspectiveHead(
            input_dim=self.backbone_feat_dim,  # Not used for projection
            img_size=img_size,
        )
        # Freeze camera head parameters (we only use it for projection)
        for param in self.camera_head.parameters():
            param.requires_grad = False
        
        # Create projection layer for features to decoder dimension (for uncertainty head)
        decoder_dim = self.cfg.MODEL.DECODER.DIM
        self.feat_to_decoder_dim = nn.Linear(self.backbone_feat_dim, decoder_dim)
        
        self.camera_type = "perspective"

    def forward(self, batch: Dict, num_samples: int = 0) -> Dict:
        """
        Forward pass through the simplified model.
        
        Args:
            batch: Dictionary containing:
                - img: [B, N, 3, H, W] or [B, 3, H, W] image tensor
                
        Returns:
            Dictionary containing pose predictions and 2D/3D keypoints
        """
        # Initialize batch indices
        self._initialize_batch(batch)
        batch_size, num_person = batch["img"].shape[:2]
        self.body_batch_idx = list(range(batch_size * num_person))
        
        # Preprocess images
        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=False,  # ResNet18 doesn't need special cropping
        )
        
        # Extract features with ResNet18
        # x: [B*N, 3, H, W]
        features = self.backbone(x)  # [B*N, 512, 1, 1] after global avg pool
        features = features.view(features.size(0), -1)  # [B*N, 512]
        
        # Predict MHR pose parameters using MLP head
        pred_pose = self.head_pose_mlp(features)  # [B*N, npose]
        
        # For uncertainty predictions, we need to project features to the decoder dimension
        # Since uncertainty head expects DECODER.DIM input
        if num_samples > 0:
            # Project features to decoder dimension for uncertainty head
            features_decoder_dim = self.feat_to_decoder_dim(features)
            uncertainty_output = self.head_pose(
                features_decoder_dim,
                full_cov=getattr(self.cfg.MODEL, "FULL_COV", True),
            )  # This gives us uncertainty predictions
        else:
            # Create dummy uncertainty if not sampling
            uncertainty_output = {
                "shape_uncertainty": torch.ones_like(pred_pose[:, 6+self.body_cont_dim:6+self.body_cont_dim+self.num_shape_comps]),
                "scale_uncertainty": torch.ones((pred_pose.shape[0], 10), device=pred_pose.device, dtype=pred_pose.dtype)
            }
        
        # Predict camera parameters
        pred_cam = self.head_camera(features)  # [B*N, 3]
        
        # Parse pose parameters (same structure as MHRHead)
        batch_size_flat = pred_pose.shape[0]
        
        # Global rotation (first 6)
        count = 6
        global_rot_6d = pred_pose[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)  # B x 3 x 3
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)  # B x 3
        global_trans = torch.zeros_like(global_rot_euler)
        
        # Body pose (next 260)
        pred_pose_cont = pred_pose[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim
        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
        # Zero-out hands and jaw (as in original)
        pred_pose_euler[:, mhr_param_hand_mask] = 0
        pred_pose_euler[:, -3:] = 0
        
        # Shape (next 45)
        pred_shape = pred_pose[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        
        # Scale (next 28)
        pred_scale = pred_pose[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        
        # Hand (next 108 = 54*2)
        pred_hand = pred_pose[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        
        # Face (next 72)
        pred_face = pred_pose[:, count : count + self.num_face_comps] * 0
        count += self.num_face_comps
        
        # Run MHR forward to get 3D keypoints and vertices
        mhr_output = self.head_pose.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            do_pcblend=True,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )
        
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = mhr_output
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        
        # Fix camera system difference
        if verts is not None:
            verts[..., [1, 2]] *= -1
        j3d[..., [1, 2]] *= -1
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1
        
        # Project 3D keypoints to 2D using camera parameters
        pose_output = self.camera_project(
            {
                "pred_keypoints_3d": j3d.reshape(batch_size_flat, -1, 3),
                "pred_cam": pred_cam,
            },
            batch,
        )
        
        # Prepare output dictionary
        output = {
            "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1),
            "global_rot": global_rot_euler,
            "body_pose": pred_pose_euler,
            "shape": pred_shape,
            "scale": pred_scale,
            "hand": pred_hand,
            "face": pred_face,
            "pred_keypoints_3d": j3d.reshape(batch_size_flat, -1, 3),
            "pred_vertices": (
                verts.reshape(batch_size_flat, -1, 3) if verts is not None else None
            ),
            "pred_joint_coords": (
                jcoords.reshape(batch_size_flat, -1, 3) if jcoords is not None else None
            ),
            "faces": self.head_pose.faces.cpu().numpy(),
            "joint_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
            "pred_cam": pred_cam,
        }
        
        # Add 2D keypoints from camera projection
        output.update(pose_output)
        
        # Convert 2D keypoints from full image to crop coordinates
        output["pred_keypoints_2d_cropped"] = self._full_to_crop(
            batch, output["pred_keypoints_2d"],
        )
        
        # Add uncertainty predictions for sampling
        output["shape_uncertainty"] = uncertainty_output.get("shape_uncertainty", torch.ones_like(pred_shape))
        output["scale_uncertainty"] = uncertainty_output.get("scale_uncertainty", torch.ones_like(pred_scale[:, [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]]))
        
        outputs = {
            "mhr": output,
            "image_embeddings": features.unsqueeze(-1).unsqueeze(-1),  # For compatibility
        }
        
        # Perform sampling if requested
        if num_samples > 0:
            outputs = self._perform_sampling(outputs, batch, num_samples)
        
        return outputs
    
    def _perform_sampling(self, outputs: Dict, batch: Dict, num_samples: int) -> Dict:
        """
        Perform sampling from uncertainty distributions for shape and scale parameters.
        Similar to SAM3DBody's sampling logic.
        """
        output_mhr = outputs['mhr']
        
        shape_params = output_mhr['shape']
        shape_uncertainty = output_mhr['shape_uncertainty']
        
        scale_params = output_mhr['scale']
        scale_uncertainty = output_mhr['scale_uncertainty']
        
        mhr_sample_verts = []
        mhr_sample_keypoints_3d = []
        for i in range(num_samples):
            # Note: shape_uncertainty and scale_uncertainty represent variance (not std dev)
            # So we need sqrt(uncertainty) to sample from N(mean, variance)
            shape_params_sample = shape_params + torch.sqrt(shape_uncertainty) * torch.randn_like(shape_params)
            
            selected_scale_comps_indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
            scale_params_sample = scale_params.clone()
            scale_params_sample[:, selected_scale_comps_indices] = (
                scale_params[:, selected_scale_comps_indices] + 
                torch.sqrt(scale_uncertainty) * torch.randn_like(scale_params[:, selected_scale_comps_indices])
            )
            
            mhr_output = self.head_pose.mhr_forward(
                scale_params=scale_params_sample,
                shape_params=shape_params_sample,
                global_trans=torch.zeros_like(output_mhr['global_rot']),
                global_rot=output_mhr['global_rot'],
                body_pose_params=output_mhr['body_pose'],
                hand_pose_params=output_mhr['hand'],
                expr_params=output_mhr['face'],
                do_pcblend=True,
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
                return_actual_scale=False,
            )
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = mhr_output
            verts[..., [1, 2]] *= -1  # Camera system difference
            j3d[..., [1, 2]] *= -1
            mhr_sample_verts.append(verts)
            mhr_sample_keypoints_3d.append(j3d[:, :70])
        
        mhr_sample_verts = torch.stack(mhr_sample_verts, dim=1)
        mhr_sample_keypoints_3d = torch.stack(mhr_sample_keypoints_3d, dim=1)
        
        outputs['mhr_samples'] = mhr_sample_verts
        outputs['kp3d_samples'] = mhr_sample_keypoints_3d
        
        # Project sampled 3D keypoints to 2D
        # Reshape samples from [B, num_samples, N, 3] to [B * num_samples, N, 3]
        B, num_samples, N, _ = mhr_sample_keypoints_3d.shape
        mhr_sample_keypoints_3d_flat = mhr_sample_keypoints_3d.view(B * num_samples, N, 3)
        
        # Get camera parameters and batch info for projection
        pred_cam = output_mhr['pred_cam']  # [B, 3]
        # Expand camera params to match flattened samples
        pred_cam_expanded = pred_cam.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
        pred_cam_flat = pred_cam_expanded.view(B * num_samples, -1)
        
        # Get batch info for projection
        bbox_center_flat = self._flatten_person(batch["bbox_center"])[self.body_batch_idx]
        bbox_center_expanded = bbox_center_flat.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
        bbox_center_flat_samples = bbox_center_expanded.view(B * num_samples, -1)
        
        bbox_scale_flat = self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0]
        bbox_scale_expanded = bbox_scale_flat.unsqueeze(1).expand(-1, num_samples).contiguous()
        bbox_scale_flat_samples = bbox_scale_expanded.view(B * num_samples)
        
        ori_img_size_flat = self._flatten_person(batch["ori_img_size"])[self.body_batch_idx]
        ori_img_size_expanded = ori_img_size_flat.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
        ori_img_size_flat_samples = ori_img_size_expanded.view(B * num_samples, -1)
        
        cam_int_flat = self._flatten_person(
            batch["cam_int"]
            .unsqueeze(1)
            .expand(-1, batch["img"].shape[1], -1, -1)
            .contiguous()
        )[self.body_batch_idx]
        cam_int_expanded = cam_int_flat.unsqueeze(1).expand(-1, num_samples, -1, -1).contiguous()
        cam_int_flat_samples = cam_int_expanded.view(B * num_samples, 3, 3)
        
        # Project to 2D (full image coordinates)
        cam_out_samples = self.camera_head.perspective_projection(
            mhr_sample_keypoints_3d_flat,
            pred_cam_flat,
            bbox_center_flat_samples,
            bbox_scale_flat_samples,
            ori_img_size_flat_samples,
            cam_int_flat_samples,
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )
        
        # Get full image 2D keypoints: [B * num_samples, N, 2]
        mhr_sample_keypoints_2d_full = cam_out_samples["pred_keypoints_2d"]
        
        # Reshape full image keypoints to [B, num_samples, N, 2]
        mhr_sample_keypoints_2d = mhr_sample_keypoints_2d_full.view(B, num_samples, N, 2)
        outputs['kp2d_samples'] = mhr_sample_keypoints_2d
        
        # Convert from full image coordinates to cropped pixel space
        # Add homogeneous coordinate for affine transformation
        mhr_sample_keypoints_2d_h = torch.cat(
            [mhr_sample_keypoints_2d_full, torch.ones_like(mhr_sample_keypoints_2d_full[..., :1])], 
            dim=-1
        )  # [B * num_samples, N, 3]
        
        # Get affine transformation for samples
        # affine_trans shape: [B*N, 2, 3] (from _flatten_person)
        affine_trans_flat = self._flatten_person(batch["affine_trans"])[self.body_batch_idx]
        affine_trans_expanded = affine_trans_flat.unsqueeze(1).expand(-1, num_samples, -1, -1).float().contiguous()
        affine_trans_flat_samples = affine_trans_expanded.view(B * num_samples, 2, 3)
        
        # Apply affine transformation to convert to cropped pixel space
        # [B * num_samples, N, 3] @ [B * num_samples, 3, 2] = [B * num_samples, N, 2]
        mhr_sample_keypoints_2d_crop = mhr_sample_keypoints_2d_h @ affine_trans_flat_samples.mT
        mhr_sample_keypoints_2d_crop = mhr_sample_keypoints_2d_crop[..., :2]
        
        # Reshape back to [B, num_samples, N, 2]
        mhr_sample_keypoints_2d_cropped = mhr_sample_keypoints_2d_crop.view(B, num_samples, N, 2)
        outputs['kp2d_samples_cropped'] = mhr_sample_keypoints_2d_cropped
        
        return outputs

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing pred_keypoints_3d and pred_cam.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        pred_cam = pose_output["pred_cam"]
        pred_keypoints_3d = pose_output["pred_keypoints_3d"]
        
        cam_out = self.camera_head.perspective_projection(
            pred_keypoints_3d,
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )
        
        return cam_out

