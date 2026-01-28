import torch
import torch.nn as nn
import pytorch_lightning as pl


class Loss(pl.LightningModule):
    def __init__(self, cfg, scale_mean, scale_comps):
        super().__init__()

        self.cfg = cfg
        self.register_buffer('scale_mean', scale_mean, persistent=False)
        self.register_buffer('scale_comps', scale_comps, persistent=False)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.kp2d_loss = nn.L1Loss(reduction='none')
        self.gaussian_nll_loss = nn.GaussianNLLLoss(reduction='mean')

        # Hand keypoint indices in MHR70: right hand 21–41, left hand 42–62
        # Total 70 keypoints; hands are indices 21–62 (42 keypoints)
        self.hand_keypoint_indices = list(range(21, 63))  # 21–62 inclusive

        # Get hand weight from config if present, otherwise default to 0.1
        hand_weight = getattr(self.cfg.LOSS, "HAND_WEIGHT", 0.1)

        # Store hand weight; full keypoint count (70 + dense) will be inferred
        # dynamically from tensors during training so we don't assume a fixed
        # dense keypoint length.
        self.hand_weight = hand_weight

    def forward(self, predictions, batch):
        loss_dict = {}

        B, N = batch['img'].shape[:2]

        pred_mhr = predictions['mhr']
        

        if self.cfg.LOSS.KP2D_WEIGHT > 0:
            # Use cropped keypoints for loss computation (in cropped pixel space, matching gt)
            if 'mhr_samples_keypoints_2d_cropped' in predictions:
                pred_kp2d_samples = predictions['mhr_samples_keypoints_2d_cropped']
            else:
                assert False
                # Fallback to full image keypoints if cropped version not available
                pred_kp2d_samples = predictions['mhr_samples_keypoints_2d']
            
            num_samples = pred_kp2d_samples.shape[1]
            
            gt_kp2d = batch['keypoints_2d']
            gt_kp2d = gt_kp2d.unsqueeze(1).expand(-1, num_samples, -1, -1)
            
            # pred_kp2d = pred_mhr['pred_keypoints_2d']
            # pred_kp2d = pred_kp2d.view(B, N, -1, 2)

            # Per-keypoint L1 loss (no reduction): [B, S, 70, 2]
            per_kp2d_loss = self.kp2d_loss(pred_kp2d_samples, gt_kp2d)
            # Average over x/y to get per-keypoint scalar loss: [B, S, 70]
            per_kp2d_loss = per_kp2d_loss.mean(dim=-1)

            # Apply keypoint weights to downweight hands.
            # kp_weights: [N_kp] -> [1, 1, N_kp] for broadcasting.
            kp_weights_expanded = self._get_kp_weights(per_kp2d_loss.shape[-1], per_kp2d_loss.device)[
                None, None, :
            ]
            weighted_kp2d_loss = per_kp2d_loss * kp_weights_expanded

            # Final scalar loss.
            loss_kp2d_samples = weighted_kp2d_loss.mean()

            loss_dict['loss_kp2d_samples'] = self.cfg.LOSS.KP2D_WEIGHT * loss_kp2d_samples

            # loss_kp2d = self.loss_fn(pred_kp2d, gt_kp2d)
            # loss_dict['loss_kp2d'] = self.cfg.LOSS.KP2D_WEIGHT * loss_kp2d


        if self.cfg.LOSS.KP3D_WEIGHT > 0:    
            gt_kp3d = batch['keypoints_3d'][..., :3]
            pred_kp3d = pred_mhr['pred_keypoints_3d']

            pred_kp3d[..., [1, 2]] *= -1

            # loss_kp3d = self.loss_fn(pred_kp3d, gt_kp3d)
            # loss_dict['loss_kp3d'] = self.cfg.LOSS.KP3D_WEIGHT * loss_kp3d

            gt_kp3d = gt_kp3d.unsqueeze(1).expand(
                -1, predictions['mhr_samples_keypoints_3d'].shape[1], -1, -1
            )
            pred_kp3d_samples = predictions['mhr_samples_keypoints_3d']
            pred_kp3d_samples[..., [1, 2]] *= -1

            # Per-keypoint MSE loss (no reduction): [B, S, 70, 3]
            per_kp3d_loss = self.mse_loss(pred_kp3d_samples, gt_kp3d)
            # Average over xyz to get per-keypoint scalar loss: [B, S, 70]
            per_kp3d_loss = per_kp3d_loss.mean(dim=-1)

            # Apply keypoint weights to downweight hands.
            kp_weights_expanded = self._get_kp_weights(per_kp3d_loss.shape[-1], per_kp3d_loss.device)[
                None, None, :
            ]
            weighted_kp3d_loss = per_kp3d_loss * kp_weights_expanded

            # Final scalar loss.
            loss_kp3d_samples = weighted_kp3d_loss.mean()
            loss_dict['loss_kp3d_samples'] = self.cfg.LOSS.KP3D_WEIGHT * loss_kp3d_samples


        if self.cfg.LOSS.SHAPE_PARAM_WEIGHT > 0:
            gt_shape_params = batch['shape_params']
            pred_shape_params = pred_mhr['shape']
            pred_shape_uncertainty = pred_mhr['shape_uncertainty']


            # Handle batch dimensions: flatten person dimension if needed
            if gt_shape_params.dim() == 3:  # [B, N, num_shape_comps]
                B, N = gt_shape_params.shape[:2]
                gt_shape_params = gt_shape_params.view(B * N, -1)
                pred_shape_params = pred_shape_params.view(B * N, -1)
                pred_shape_uncertainty = pred_shape_uncertainty.view(B * N, -1)
            elif gt_shape_params.dim() == 2:  # Already flattened [B*N, num_shape_comps]
                pass
            else:
                # Reshape to [B*N, num_shape_comps]
                gt_shape_params = gt_shape_params.view(-1, gt_shape_params.shape[-1])
                pred_shape_params = pred_shape_params.view(-1, pred_shape_params.shape[-1])
                pred_shape_uncertainty = pred_shape_uncertainty.view(-1, pred_shape_uncertainty.shape[-1])
            
            # Use Gaussian NLL loss: assumes target ~ N(pred, uncertainty)
            # uncertainty is already positive exp(...) and represents variance
            loss_shape_params = self.gaussian_nll_loss(
                pred_shape_params, 
                gt_shape_params, 
                pred_shape_uncertainty
            )
            loss_dict['loss_shape_params'] = self.cfg.LOSS.SHAPE_PARAM_WEIGHT * loss_shape_params

        if self.cfg.LOSS.SCALE_PARAM_WEIGHT > 0:

            gt_scale_params = batch['scale_params']
            pred_scale_params = pred_mhr['scale']
            pred_scale_uncertainty = pred_mhr['scale_uncertainty']

            pred_scale_params = pred_scale_params @ self.scale_comps

            # Only compute loss for selected scale component indices
            selected_scale_comps_indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]

            # Handle batch dimensions: flatten person dimension if needed
            if gt_scale_params.dim() == 3:  # [B, N, num_scale_comps]
                B, N = gt_scale_params.shape[:2]
                gt_scale_params = gt_scale_params.view(B * N, -1)
                pred_scale_params = pred_scale_params.view(B * N, -1)
                pred_scale_uncertainty = pred_scale_uncertainty.view(B * N, -1)
            elif gt_scale_params.dim() == 2:  # Already flattened [B*N, num_scale_comps]
                pass
            else:
                # Reshape to [B*N, num_scale_comps]
                gt_scale_params = gt_scale_params.view(-1, gt_scale_params.shape[-1])
                pred_scale_params = pred_scale_params.view(-1, pred_scale_params.shape[-1])
                pred_scale_uncertainty = pred_scale_uncertainty.view(-1, pred_scale_uncertainty.shape[-1])
            
            # Extract only the selected indices for loss computation
            gt_scale_params_selected = gt_scale_params[:, selected_scale_comps_indices]
            pred_scale_params_selected = pred_scale_params[:, selected_scale_comps_indices]
            
            # Use Gaussian NLL loss: assumes target ~ N(pred, uncertainty)
            # uncertainty is already positive exp(...) and represents variance
            # pred_scale_uncertainty already has shape [B*N, num_selected_scales]
            loss_scale_params = self.gaussian_nll_loss(
                pred_scale_params_selected, 
                gt_scale_params_selected, 
                pred_scale_uncertainty
            )
            loss_dict['loss_scale_params'] = self.cfg.LOSS.SCALE_PARAM_WEIGHT * loss_scale_params
            
        assert 'total_loss' not in loss_dict
        loss_dict['total_loss'] = sum(v for k, v in loss_dict.items() if k != 'total_loss')


        # for k, v in loss_dict.items():
        #     print(f'{k}: {v.item():.3f}', end=' ')
        # import ipdb; ipdb.set_trace()

        return loss_dict

    def _get_kp_weights(self, num_kp: int, device: torch.device) -> torch.Tensor:
        """
        Construct per-keypoint weights of length `num_kp`.
        - First 70 keypoints follow the canonical MHR70 layout; hand indices
          (21–62) get `self.hand_weight`.
        - Any additional dense keypoints beyond index 69 get weight 1.0.
        """
        kp_weights = torch.ones(num_kp, device=device)

        # Apply hand weight only where indices exist within current keypoint set
        for idx in self.hand_keypoint_indices:
            if idx < num_kp:
                kp_weights[idx] = self.hand_weight

        return kp_weights