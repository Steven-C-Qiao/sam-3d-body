# Copyright (c) Meta Platforms, Inc. and affiliates.

import pickle
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..modules.transformer import build_norm_layer, TransformerDecoderLayer

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class PromptableDecoder(nn.Module):
    """Cross-attention based Transformer decoder with prompts input.

    Args:
        token_dims (int): The dimension of input pose tokens.
        prompt_dims (int): The dimension of input prompt tokens.
        context_dims (int): The dimension of image context features.
        dims (int): The projected dimension of all tokens in the decoder.
        depth (int): The number of layers for Transformer decoder.
        num_heads (int): The number of heads for multi-head attention.
        head_dims (int): The dimension of each head.
        mlp_dims (int): The dimension of hidden layers in MLP.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_layer (nn.Module, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        enable_twoway (bool): Whether to enable two-way Transformer (used in SAM).
        repeat_pe (bool): Whether to re-add PE at each layer (used in SAM)
    """

    def __init__(
        self,
        dims: int,
        context_dims: int,
        depth: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        act_layer: nn.Module = nn.GELU,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = False,
        repeat_pe: bool = False,
        frozen: bool = False,
        do_interm_preds: bool = False,
        do_keypoint_tokens: bool = False,
        keypoint_token_update: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TransformerDecoderLayer(
                    token_dims=dims,
                    context_dims=context_dims,
                    num_heads=num_heads,
                    head_dims=head_dims,
                    mlp_dims=mlp_dims,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    ffn_type=ffn_type,
                    act_layer=act_layer,
                    norm_cfg=norm_cfg,
                    enable_twoway=enable_twoway,
                    repeat_pe=repeat_pe,
                    skip_first_pe=(i == 0),
                )
            )

        self.norm_final = build_norm_layer(norm_cfg, dims)
        self.do_interm_preds = do_interm_preds
        self.do_keypoint_tokens = do_keypoint_tokens
        self.keypoint_token_update = keypoint_token_update

        self.frozen = frozen
        self._freeze_stages()

        # Store LoRA config for later application
        # We apply LoRA after initialization to avoid issues with PEFT wrapping
        self._use_lora = use_lora
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._lora_target_modules = lora_target_modules
        
        # Create separate LoRA layers if enabled (keeping original layers frozen)
        self.lora_layers = None
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT library is required for LoRA support. "
                    "Please install it with: pip install peft"
                )
            self._create_lora_layers()

    def forward(
        self,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor] = None,
        image_augment: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        channel_first: bool = True,
        token_to_pose_output_fn=None,
        token_to_uncertainty_output_fn=None,
        keypoint_token_update_fn=None,
        hand_embeddings=None,
        hand_augment=None,
    ):
        """
        Args:
            token_embedding: [B, N, C]
            image_embedding: [B, C, H, W]
        
        Returns:
            If LoRA is enabled: (original_output, lora_output) or 
                                (original_output, all_pose_outputs_orig), (lora_output, all_pose_outputs_lora)
            If LoRA is disabled: original_output or (original_output, all_pose_outputs)
        """
        if channel_first:
            image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
            if image_augment is not None:
                image_augment = image_augment.flatten(2).permute(0, 2, 1)
            if hand_embeddings is not None:
                hand_embeddings = hand_embeddings.flatten(2).permute(0, 2, 1)
                hand_augment = hand_augment.flatten(2).permute(0, 2, 1)
                if len(hand_augment) == 1:
                    # inflate batch dimension
                    assert len(hand_augment.shape) == 3
                    hand_augment = hand_augment.repeat(len(hand_embeddings), 1, 1)

        # Run original frozen decoder
        orig_output = self._forward_path(
            self.layers,
            token_embedding,
            image_embedding,
            token_augment,
            image_augment,
            token_mask,
            hand_embeddings,
            hand_augment,
            token_to_pose_output_fn,
            keypoint_token_update_fn,
        )

        # Run LoRA path if enabled
        if self._use_lora and self.lora_layers is not None:
            lora_output = self._forward_path(
                self.lora_layers,
                token_embedding,
                image_embedding,
                token_augment,
                image_augment,
                token_mask,
                hand_embeddings,
                hand_augment,
                token_to_uncertainty_output_fn,
                keypoint_token_update_fn,
                lora_path=True,
            )
            return orig_output, lora_output
        else:
            return orig_output

    def _forward_path(
        self,
        layers,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor],
        image_augment: Optional[torch.Tensor],
        token_mask: Optional[torch.Tensor],
        hand_embeddings: Optional[torch.Tensor],
        hand_augment: Optional[torch.Tensor],
        token_to_output_fn,
        keypoint_token_update_fn,
        lora_path: bool = False,
    ):
        """Helper method to run forward through a set of layers."""
        if self.do_interm_preds:
            assert token_to_output_fn is not None
            all_pose_outputs = []

        for layer_idx, layer in enumerate(layers):
            if hand_embeddings is None:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    image_embedding,
                    token_augment,
                    image_augment,
                    token_mask,
                )
            else:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    torch.cat([image_embedding, hand_embeddings], dim=1),
                    token_augment,
                    torch.cat([image_augment, hand_augment], dim=1),
                    token_mask,
                )
                image_embedding = image_embedding[:, : image_augment.shape[1]]

            if lora_path:
                pass

            elif self.do_interm_preds and layer_idx < len(layers) - 1:
                curr_pose_output = token_to_output_fn(
                    self.norm_final(token_embedding),
                    prev_pose_output=(
                        all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                    ),
                    layer_idx=layer_idx,
                )
                all_pose_outputs.append(curr_pose_output)

                if self.keypoint_token_update:
                    assert keypoint_token_update_fn is not None
                    token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                        token_embedding, token_augment, curr_pose_output, layer_idx
                    )

        out = self.norm_final(token_embedding)

        if lora_path:
            uncertainty_output = token_to_output_fn(out)
            return uncertainty_output 

        if self.do_interm_preds:
            curr_pose_output = token_to_output_fn(
                out,
                prev_pose_output=(
                    all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                ),
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr_pose_output)
            return out, all_pose_outputs
        else:
            return out

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen:
            for layer in self.layers:
                layer.eval()
            self.norm_final.eval()
            for param in self.parameters():
                param.requires_grad = False

    def _create_lora_layers(self):
        """Create separate LoRA-wrapped layers while keeping original layers frozen."""
        if self._lora_target_modules is None:
            # Default target modules for transformer decoder
            # These will match Linear layers in attention and FFN modules
            # PEFT matches by module name, so we need to match the actual names
            # in the TransformerDecoderLayer structure
            target_modules = ["q_proj", "k_proj", "v_proj", "proj"]
        else:
            target_modules = self._lora_target_modules

        # Create LoRA config
        lora_config = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            target_modules=target_modules,
            lora_dropout=self._lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Create LoRA-wrapped copies of each layer.
        # We deep copy the layers to avoid modifying the original frozen layers.
        # Important: we only keep the *base_model* (original module type with LoRA
        # weights injected), not the PeftModel wrapper, so the forward signature
        # stays the same and no HF-style kwargs (e.g. input_ids) are introduced.
        import copy
        self.lora_layers = nn.ModuleList()
        for layer in self.layers:
            # Deep copy the layer to create a separate instance
            lora_layer = copy.deepcopy(layer)
            # Let PEFT inject LoRA adapters into the copied layer
            peft_wrapped = get_peft_model(lora_layer, lora_config)
            # Extract the mutated base model (a TransformerDecoderLayer with LoRA)
            self.lora_layers.append(peft_wrapped.base_model)

    def train(self, mode=True):
        """
        Convert the model into training mode.
        (not called by lightning in trainer.fit() actually)
        """
        super().train(mode)
        self._freeze_stages()
