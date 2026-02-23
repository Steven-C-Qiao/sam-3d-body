# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_3d_body.models.decoders.prompt_encoder import PositionEmbeddingRandom
from sam_3d_body.models.modules.mhr_utils import (
    fix_wrist_euler,
    rotation_angle_difference,
)
from sam_3d_body.utils import recursive_to
from sam_3d_body.utils.logging import get_pylogger

from ..backbones import create_backbone
from ..decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from ..heads import build_head
from ..modules.camera_embed import CameraEncoder
from ..modules.transformer import FFN, MLP
from ..modules import rot6d_to_rotmat

from .base_model import BaseModel

from sam_3d_body.models.sampling import gen_samples

logger = get_pylogger(__name__)

# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "mhr70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
KEY_RIGHT_HAND = list(range(21, 42))
# fmt: on


class SAM3DBody(BaseModel):
    # fmt: off
    pelvis_idx = [9, 10]  # left_hip, right_hip

    def _initialze_model(self):
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        self.sample_shape = self.cfg.MODEL.SAMPLE_SHAPE
        self.sample_scale = self.cfg.MODEL.SAMPLE_SCALE
        self.sample_pose = self.cfg.MODEL.SAMPLE_POSE
        self.full_cov = self.cfg.MODEL.FULL_COV
        self.use_uncertainty_token = self.cfg.MODEL.UNCERTAINTY_TOKEN

        # Create backbone feature extractor for human crops
        self.backbone = create_backbone(self.cfg.MODEL.BACKBONE.TYPE, self.cfg)

        # Create header for pose estimation output
        self.head_pose = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        self.head_pose.hand_pose_comps_ori = nn.Parameter(
            self.head_pose.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose.hand_pose_comps.data).float()
        )

        # Initialize pose token with learnable params
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        self.init_pose = nn.Embedding(1, self.head_pose.npose)

        # Define header for hand pose estimation
        self.head_pose_hand = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(
            self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose_hand.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        )
        self.init_pose_hand = nn.Embedding(1, self.head_pose_hand.npose)

        self.head_camera = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE)
        self.init_camera = nn.Embedding(1, self.head_camera.ncam)
        nn.init.zeros_(self.init_camera.weight)

        self.head_camera_hand = build_head(
            self.cfg,
            self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=self.cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
        )
        self.init_camera_hand = nn.Embedding(1, self.head_camera_hand.ncam)
        nn.init.zeros_(self.init_camera_hand.weight)

        self.camera_type = "perspective"

        # Support conditioned information for decoder
        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        self.init_to_token_mhr = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )
        self.init_to_token_mhr_hand = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr_hand = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )

        # Create prompt encoder
        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,  # need to match backbone dims for PE
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
            )
            self.prompt_to_token = nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=(
                    KEY_BODY
                    if not self.cfg.MODEL.PROMPT_ENCODER.get("SAMPLE_HAND", False)
                    else KEY_RIGHT_HAND
                ),
            )
            # To keep track of prompting history
            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False

        # Create promptable decoder
        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        # shared config for the two decoders
        self.decoder_hand = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

        # Manually convert the torso of the model to fp16.
        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )
        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )

        self.keypoint_embedding_idxs = list(range(70))
        self.keypoint_embedding = nn.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = nn.Embedding(
            len(self.keypoint_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = nn.Embedding(
                2, self.cfg.MODEL.DECODER.DIM
            )  # for two hands
            # decice if there is left or right hand inside the image
            self.hand_cls_embed = nn.Linear(self.cfg.MODEL.DECODER.DIM, 2)
            self.bbox_embed = MLP(
                self.cfg.MODEL.DECODER.DIM, self.cfg.MODEL.DECODER.DIM, 4, 3
            )

        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_feat_linear_hand = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )

        # Do all KPS
        self.keypoint3d_embedding_idxs = list(range(70))
        self.keypoint3d_embedding = nn.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )

        # Assume always do full body for the hand decoder
        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = nn.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint3d_posemb_linear_hand = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )

        # Learnable uncertainty token for better uncertainty prediction (optional)
        # This token attends to all other tokens and image features but is only used for uncertainty
        if self.use_uncertainty_token:
            self.uncert_token = nn.Parameter(
                torch.zeros(1, 1, self.cfg.MODEL.DECODER.DIM)
            )
            nn.init.normal_(self.uncert_token, std=0.02)

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        num_person = batch["img"].shape[1]

        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, [0]]

            focal_length = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, num_person, -1, -1)
                .contiguous()
            )[:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, num_person, -1, -1)
                    .contiguous()
                )[:, [0, 1], [2, 2]]
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )  # [-1, 1]
            condition_info[:, 2] = condition_info[:, 2] / focal_length  # [-1, 1]
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1

        # Optionally insert uncertainty token right after pose token (at index 1)
        uncert_token_idx = None
        if getattr(self, "use_uncertainty_token", False):
            uncert_token_emb = self.uncert_token.expand(
                batch_size, -1, -1
            )  # B x 1 x D
            token_embeddings = torch.cat(
                [token_embeddings, uncert_token_emb], dim=1
            )  # B x 2 x D (pose token + uncertainty token)
            uncert_token_idx = 1  # Index of uncertainty token in the sequence

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_mhr(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)

            image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            # Tokens currently: [pose] or [pose, uncert], then we concatenate prev & prompt.
            # Compute where the newly added prev and prompt tokens live in the sequence.
            start_new = token_embeddings.shape[1] - (
                prev_embeddings.shape[1] + prompt_embeddings.shape[1]
            )
            prev_start = start_new
            prompt_start = start_new + prev_embeddings.shape[1]
            token_augment[:, prev_start : prev_start + prev_embeddings.shape[1]] = (
                prev_embeddings
            )
            token_augment[
                :, prompt_start : prompt_start + prompt_embeddings.shape[1]
            ] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024

        # We're doing intermediate model predictions
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            # Get the pose token (index 0) and (optionally) the uncertainty token
            pose_token = tokens[:, 0]
            uncert_token = (
                tokens[:, uncert_token_idx]
                if uncert_token_idx is not None
                else None
            )

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # Get pose outputs, passing uncertainty token for uncertainty predictions
            pose_output = self.head_pose(
                pose_token,
                prev_pose,
                full_cov=self.full_cov,
                uncert_token=uncert_token,
            )
            # Get Camera Translation
            if hasattr(self, "head_camera"):
                pred_cam = self.head_camera(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam
            # Run camera projection
            pose_output = self.camera_project(pose_output, batch)

            # Get 2D KPS in crop
            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.body_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn

        # Combine the 2D and 3D functionse
        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        tokens_output, pose_output = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        # Extract pose token (index 0) from final tokens
        pose_token = tokens_output[:, 0]

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                tokens_output[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    @torch.no_grad()
    def _get_keypoint_prompt(self, batch, pred_keypoints_2d, force_dummy=False):
        if self.camera_type == "perspective":
            pred_keypoints_2d = self._full_to_crop(batch, pred_keypoints_2d)

        gt_keypoints_2d = self._flatten_person(batch["keypoints_2d"]).clone()

        keypoint_prompt = self.keypoint_prompt_sampler.sample(
            gt_keypoints_2d,
            pred_keypoints_2d,
            is_train=self.training,
            force_dummy=force_dummy,
        )
        return keypoint_prompt

    def _get_mask_prompt(self, batch, image_embeddings):
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
        ]:
            # ViT backbone assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            # for x2 resolution
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera"):
            head_camera = self.head_camera
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
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

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
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
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output


    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(B, N, 1, 1, 1)
            .cuda()
        )  # B x N x H x W x 2
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )

        # Subtract out center & normalize to be rays
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )  # This is B x num_person x 2 x H x W

    def forward_pose_branch(self, batch: Dict) -> Dict:
        """Run a forward pass for the crop-image (pose) branch."""
        batch_size, num_person = batch["img"].shape[:2]

        # Forward backbone encoder
        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                ]
            ),
        )

        # Optionally get ray conditioining
        ray_cond = self.get_ray_condition(batch)  # This is B x num_person x 2 x H x W
        ray_cond = self._flatten_person(ray_cond)
        batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
        ray_cond = None

        image_embeddings = self.backbone(
            x.type(self.backbone_dtype), extra_embed=ray_cond
        )  # (B, C, H, W)

        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)


        # Mask condition if available
        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            # v1: non-iterative mask conditioning
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError


        # Prepare input for promptable decoder
        condition_info = self._get_decoder_condition(batch)

        # Initial estimate with a dummy prompt
        keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
        keypoints_prompt[:, :, -1] = -2

        # Forward promptable decoder to get updated pose tokens and regression output
        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            pose_output = pose_output[-1]

        output = {
            "mhr": pose_output,  # mhr prediction output
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }
        return output
    # fmt: on

    def forward(self, batch: Dict, num_samples: int = 0) -> Dict:
        """
        Forward pass for training.
        Uses a single forward step through the pose branch (body decoder).
        For inference with refinement, use run_inference() instead.
        """
        # Initialize batch indices for body decoder
        self._initialize_batch(batch)
        B, N = batch["img"].shape[:2]
        self.hand_batch_idx = []
        self.body_batch_idx = list(range(B * N))

        outputs = self.forward_pose_branch(batch)

        if num_samples > 0:
            output_mhr = outputs["mhr"]

            global_rot_6d_mean = output_mhr["pred_pose_raw"][:, :6]
            global_rot_mat_mean = rot6d_to_rotmat(global_rot_6d_mean)
            global_rot_euler_mean = roma.rotmat_to_euler("ZYX", global_rot_mat_mean)
            global_rot_euler_mean = global_rot_euler_mean.repeat_interleave(
                num_samples, dim=0
            )
            global_trans = torch.zeros_like(global_rot_euler_mean)

            shape_samples, scale_samples, pose_samples = gen_samples(
                output_mhr,
                num_samples,
                sample_pose=self.sample_pose,
                full_cov=self.full_cov,
            )
            shape_samples = shape_samples.view(-1, shape_samples.shape[-1])
            scale_samples = scale_samples.view(-1, scale_samples.shape[-1])
            pose_samples = pose_samples.view(-1, pose_samples.shape[-1])

            mhr_output = self.head_pose.mhr_forward(
                scale_params=torch.zeros_like(scale_samples),
                shape_params=shape_samples,
                global_trans=global_trans,
                global_rot=global_rot_euler_mean,
                body_pose_params=pose_samples,
                hand_pose_params=output_mhr["hand"].repeat_interleave(
                    num_samples, dim=0
                ),
                expr_params=output_mhr["face"].repeat_interleave(num_samples, dim=0),
                do_pcblend=True,
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
                scale_offsets=scale_samples,
            )
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = mhr_output
            verts[..., [1, 2]] *= -1  # Camera system difference
            j3d[..., [1, 2]] *= -1
            j3d = j3d[:, :70]
            jcoords[..., [1, 2]] *= -1
            if (
                hasattr(self, "mhr_dense_kp_indices")
                and self.mhr_dense_kp_indices is not None
            ):
                dense_kp3d = verts[:, self.mhr_dense_kp_indices]
                j3d = torch.cat([j3d, dense_kp3d], dim=1)

            outputs["verts_samples"] = verts.view(B, num_samples, -1, 3)
            outputs["kp3d_samples"] = j3d.view(B, num_samples, -1, 3)
            outputs["j3d_samples"] = jcoords.view(B, num_samples, -1, 3)

            j3d_flat = j3d.view(B * num_samples, -1, 3)
            jcoords_flat = jcoords.view(B * num_samples, -1, 3)

            pred_cam_expanded = output_mhr["pred_cam"].repeat_interleave(
                num_samples, dim=0
            )
            bbox_center_expanded = self._flatten_person(batch["bbox_center"])[
                self.body_batch_idx
            ].repeat_interleave(num_samples, dim=0)
            bbox_scale_expanded = self._flatten_person(batch["bbox_scale"])[
                self.body_batch_idx, 0
            ].repeat_interleave(num_samples, dim=0)
            ori_img_size_expanded = self._flatten_person(batch["ori_img_size"])[
                self.body_batch_idx
            ].repeat_interleave(num_samples, dim=0)

            cam_int_flat = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx]
            cam_int_expanded = cam_int_flat.repeat_interleave(num_samples, dim=0)

            # Project to 2D (full image coordinates)
            kp2d_samples = self.head_camera.perspective_projection(
                j3d_flat,
                pred_cam_expanded,
                bbox_center_expanded,
                bbox_scale_expanded,
                ori_img_size_expanded,
                cam_int_expanded,
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )["pred_keypoints_2d"]
            outputs["kp2d_samples"] = kp2d_samples.view(B, num_samples, -1, 2)

            kp2d_samples_h = torch.cat(
                [
                    kp2d_samples,
                    torch.ones_like(kp2d_samples[..., :1]),
                ],
                dim=-1,
            )  # [B * num_samples, N, 3]

            affine = self._flatten_person(batch["affine_trans"])[
                self.body_batch_idx
            ].repeat_interleave(num_samples, dim=0)
            img_size = self._flatten_person(batch["img_size"])[
                self.body_batch_idx
            ].repeat_interleave(num_samples, dim=0)

            kp2d_samples_crop = kp2d_samples_h @ affine.mT
            # kp2d_samples_crop = kp2d_samples_crop[..., :2]
            kp2d_samples_crop = (kp2d_samples_crop / img_size - 0.5).view(
                B, num_samples, -1, 2
            )
            outputs["kp2d_samples_cropped"] = kp2d_samples_crop

            jcoords_samples = self.head_camera.perspective_projection(
                jcoords_flat,
                pred_cam_expanded,
                bbox_center_expanded,
                bbox_scale_expanded,
                ori_img_size_expanded,
                cam_int_expanded,
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )["pred_keypoints_2d"]

            outputs["j2d_samples"] = jcoords_samples.view(B, num_samples, -1, 2)

            jcoords_samples_h = torch.cat(
                [jcoords_samples, torch.ones_like(jcoords_samples[..., :1])],
                dim=-1,
            )
            jcoords_samples_crop = jcoords_samples_h @ affine.mT
            # jcoords_samples_crop = jcoords_samples_crop[..., :2]
            jcoords_samples_crop = jcoords_samples_crop / img_size - 0.5
            outputs["j2d_samples_cropped"] = jcoords_samples_crop.view(
                B, num_samples, -1, 2
            )

        return outputs

    # fmt: off
    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["mhr"]  # body-only output
        # Use previous estimate as initialization
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),  # (B, 6)
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                dim=-1,
            )

        tokens_output, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,  # not recurring previous estimate
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_output[-1]

        output.update({"mhr": pose_output})
        return output, keypoint_prompt

    def _get_hand_box(self, pose_output, batch):
        """Get hand bbox from the hand detector"""
        pred_left_hand_box = (
            pose_output["mhr"]["hand_box"][:, 0].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )
        pred_right_hand_box = (
            pose_output["mhr"]["hand_box"][:, 1].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )

        # Change boxes into squares
        batch["left_center"] = pred_left_hand_box[:, :2]
        batch["left_scale"] = (
            pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )
        batch["right_center"] = pred_right_hand_box[:, :2]
        batch["right_scale"] = (
            pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )

        # Crop to full. batch["affine_trans"] is full-to-crop, right application
        batch["left_scale"] = (
            batch["left_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["right_scale"] = (
            batch["right_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["left_center"] = (
            batch["left_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        batch["right_center"] = (
            batch["right_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]

        left_xyxy = np.concatenate(
            [
                (
                    batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )
        right_xyxy = np.concatenate(
            [
                (
                    batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )

        return left_xyxy, right_xyxy

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs_hand
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs_hand
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs_hand]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx
    # fmt: on
