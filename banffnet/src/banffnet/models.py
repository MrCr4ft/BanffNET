"""Noisy-OR and gated-ratio MIL heads used by BanffNET.

``MODELS`` maps YAML ``model_type`` values to the corresponding head classes.
"""

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn


def logistic_sum_model(
    probabilities: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Smooth approximation of the noisy-OR bag probability.

    Replaces the strict ``1 - ∏(1 - p_i)`` with a sigmoid of the
    (re-centred) sum of instance probabilities — gentler gradients and
    less prone to saturation when the bag contains many easy positives.
    """
    sum_probs = torch.sum(probabilities, dim=0)
    return 1 / (1 + torch.exp(-beta * (sum_probs - gamma)))


def _init_linear_weights(module: nn.Module) -> None:
    """Xavier-uniform init for every :class:`nn.Linear` under ``module``."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class LinearNoisyOR(nn.Module):
    """Linear classifier + noisy-OR aggregation.

    Each instance is scored by a single linear layer + sigmoid. The bag
    probability is ``1 - ∏(1 - p_i)`` (or a numerically stabilised
    log-space variant). At inference an optional logistic-sum
    approximation is used.

    Parameters
    ----------
    input_features_size
        Patch feature dimension. Must match the backbone and checkpoint;
        UNI ViT-L/16 features have 1024 dimensions.
    output_features_size
        Number of output classes (typically 1 for binary attributes).
    bias
        Whether the classifier head has a bias.
    normalize_by_bag_size
        Divide the log-sum by the bag size before exponentiating.
        Requires ``apply_numerical_stabilization=True``.
    apply_numerical_stabilization
        Compute the noisy-OR in log space.
    logistic_sum_inference, inference_beta, inference_gamma
        At inference time, optionally replace the noisy-OR with the
        logistic-sum approximation parameterised by
        ``(inference_beta, inference_gamma)``.
    """

    def __init__(
        self,
        input_features_size: int = 2048,
        output_features_size: int = 1,
        bias: bool = True,
        normalize_by_bag_size: bool = False,
        apply_numerical_stabilization: bool = False,
        logistic_sum_inference: bool = True,
        inference_beta: float = 1.0,
        inference_gamma: float = 2.0,
    ):
        super().__init__()
        self.classifier = nn.Linear(input_features_size, output_features_size, bias=bias)

        if normalize_by_bag_size and not apply_numerical_stabilization:
            raise ValueError("normalize_by_bag_size requires apply_numerical_stabilization=True")
        self.normalize_by_bag_size = normalize_by_bag_size
        self.apply_numerical_stabilization = apply_numerical_stabilization

        self.logistic_sum_inference = logistic_sum_inference
        self.inference_beta = inference_beta
        self.inference_gamma = inference_gamma
        _init_linear_weights(self)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Strict noisy-OR aggregation — used during training."""
        p = torch.sigmoid(self.classifier(h))
        p = torch.nan_to_num(p, nan=0.0)

        if self.apply_numerical_stabilization:
            log_complement = torch.log(1 - p + 1e-10)
            sum_log_complement = torch.sum(log_complement, dim=0)
            if self.normalize_by_bag_size:
                sum_log_complement = sum_log_complement / p.shape[0]
            return 1 - torch.exp(sum_log_complement)
        return 1 - torch.prod(1 - p, dim=0)

    def infer(self, h: torch.Tensor) -> torch.Tensor:
        """Bag-level inference: logistic-sum approximation if configured,
        otherwise the strict noisy-OR (:meth:`forward`).
        """
        if self.logistic_sum_inference:
            p = torch.sigmoid(self.classifier(h))
            p = torch.nan_to_num(p, nan=0.0)
            return logistic_sum_model(p, self.inference_beta, self.inference_gamma)
        return self.forward(h)

    def infer_patch_level(self, h: torch.Tensor) -> torch.Tensor:
        """Per-patch instance probabilities, shape ``[N_patches, 1]``."""
        p = torch.sigmoid(self.classifier(h))
        return torch.nan_to_num(p, nan=0.0)


class LinearGatedRatio(nn.Module):
    """Gated-ratio aggregation.

    Two linear heads on the same input:

    * a ``gate`` that produces a per-instance gating coefficient in ``(0, 1)``,
    * a ``classifier`` that produces per-instance class probabilities on the
      gated features.

    The bag probability is the gated sum / sum of gates (a learned weighted
    mean).
    """

    def __init__(
        self,
        input_features_size: int = 2048,
        output_features_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.gate = nn.Linear(input_features_size, 1, bias=bias)
        self.classifier = nn.Linear(input_features_size, output_features_size, bias=bias)
        _init_linear_weights(self)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(h))
        gate = torch.nan_to_num(gate, nan=0.0)
        if mask is not None:
            gate = mask.unsqueeze(-1) * gate

        gated = gate * h
        p = torch.sigmoid(self.classifier(gated))
        p = torch.nan_to_num(p, nan=0.0)

        bag_p = torch.sum(p, dim=1) / (torch.sum(gate, dim=1) + 1e-10)
        return torch.clamp(bag_p, 0, 1).unsqueeze(-1)  # [N, 1]

    def infer(self, h: torch.Tensor) -> torch.Tensor:
        """Bag-level inference for a single instance (no batch dim)."""
        return self.forward(h.unsqueeze(0)).squeeze(0)

    def infer_patch_level(self, h: torch.Tensor) -> torch.Tensor:
        """Per-patch gated probabilities, shape ``[N_patches, 1]``."""
        gate = torch.sigmoid(self.gate(h))
        gate = torch.nan_to_num(gate, nan=0.0)
        p = torch.sigmoid(self.classifier(gate * h))
        p = torch.nan_to_num(p, nan=0.0)
        return p * gate


#: Map YAML ``model_type`` strings to the head classes.
MODELS: Dict[str, Type[nn.Module]] = {
    "linear_noisy_or": LinearNoisyOR,
    "linear_gated_ratio": LinearGatedRatio,
}
