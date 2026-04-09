import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm


class ClampedAffineCouplingTransform(AffineCouplingTransform):
    """Affine coupling with explicit log-scale clamp."""

    def _scale_and_shift(self, transform_params):
        shift = transform_params[:, : self.num_transform_features, ...]
        raw_log_scale = transform_params[:, self.num_transform_features :, ...]
        log_scale = torch.tanh(raw_log_scale) * 2.0
        scale = torch.exp(log_scale)
        return scale, shift


class ConditionalGlowAffine(Flow):
    """Conditional Glow variant with affine coupling layers."""

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        activation=F.relu,
        dropout_probability=0.5,
        context_features=None,
        batch_norm_within_layers=True,
    ):
        coupling_constructor = ClampedAffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                context_features=context_features,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            layers.append(ActNorm(features=features))
            layers.append(LULinear(features=features))
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            mask *= -1
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
