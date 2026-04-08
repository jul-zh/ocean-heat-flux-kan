from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMAELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(1)
        diff = torch.abs(pred - target) * mask
        return diff.sum() / (mask.sum() * pred.shape[1] + 1e-8)


class MaskedMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(1)
        diff = ((pred - target) ** 2) * mask
        return diff.sum() / (mask.sum() * pred.shape[1] + 1e-8)


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mae = MaskedMAELoss()
        self.mse = MaskedMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mae(pred, target, mask) + (1 - self.alpha) * self.mse(pred, target, mask)


class MaskedGaussianNLLLoss(nn.Module):
    def __init__(self, aux_mae_weight: float = 0.1):
        super().__init__()
        self.aux_mae_weight = aux_mae_weight
        self.mae = MaskedMAELoss()

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(1)
        var = sigma.pow(2).clamp_min(1e-6)
        nll = 0.5 * (torch.log(2 * torch.tensor(math.pi, device=target.device) * var) + ((target - mu) ** 2) / var)
        nll = nll * mask
        nll = nll.sum() / (mask.sum() * mu.shape[1] + 1e-8)
        aux = self.mae(mu, target, mask.squeeze(1))
        return nll + self.aux_mae_weight * aux


def get_loss(head_name: str, config: dict) -> nn.Module:
    if head_name.endswith("_gauss"):
        return MaskedGaussianNLLLoss(aux_mae_weight=config["loss"]["gaussian_aux_mae_weight"])
    return CombinedLoss(alpha=config["loss"]["deterministic_alpha"])


def is_probabilistic_head(head_name: str) -> bool:
    return head_name.endswith("_gauss")


def unpack_prediction(pred, probabilistic: bool):
    if probabilistic:
        return pred["mu"], pred["sigma"]
    return pred["pred"]
