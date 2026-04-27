import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
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


class ScaledStandardNormal(StandardNormal):
    """N(0, std² · I) — base distribution with isotropic non-unit variance.

    The full distribution-space behaviour stays a centered isotropic Gaussian;
    only the radius is scaled. log_prob and sampling are derived from the
    standard normal via the change-of-variables y = z / std.
    """

    def __init__(self, shape, std=1.0):
        super().__init__(shape)
        self.register_buffer(
            "_base_std", torch.tensor(float(std), dtype=torch.float64), persistent=False,
        )
        d = float(int(torch.tensor(shape).prod().item()))
        self.register_buffer(
            "_base_std_log_det",
            torch.tensor(d * math.log(float(std)), dtype=torch.float64),
            persistent=False,
        )

    def _log_prob(self, inputs, context):
        std = self._base_std.to(inputs.dtype)
        scaled = inputs / std
        return super()._log_prob(scaled, context) - self._base_std_log_det.to(inputs.dtype)

    def _sample(self, num_samples, context):
        z = super()._sample(num_samples, context)
        return z * self._base_std.to(z.dtype)


def _make_base_distribution(features, base_std):
    if base_std == 1.0:
        return StandardNormal([features])
    return ScaledStandardNormal([features], std=base_std)


class ConditionalGlowUnclampedAffine(Flow):
    """Conditional Glow variant with unclamped affine coupling layers."""

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
        base_std=1.0,
    ):
        coupling_constructor = AffineCouplingTransform

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
            distribution=_make_base_distribution(features, base_std),
        )


class ConditionalGlowAffine(Flow):
    """Conditional Glow variant with clamped affine coupling layers."""

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
        base_std=1.0,
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
            distribution=_make_base_distribution(features, base_std),
        )


class ConditionalGlowSpline(Flow):
    """Conditional Glow with piecewise rational-quadratic spline coupling (NSF)."""

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
        num_bins=10,
        tails="linear",
        tail_bound=3.0,
        base_std=1.0,
    ):
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
            transform = PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_resnet,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
            )
            mask *= -1
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=_make_base_distribution(features, base_std),
        )
