from __future__ import annotations

import math

import torch
import torch.nn as nn


class MaskedMAELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(1)
        diff = torch.abs(pred - target) * mask
        return diff.sum() / (mask.sum() * pred.shape[1] + 1e-8)


class MaskedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(1)
        err = torch.abs(pred - target)
        quad = torch.minimum(err, torch.tensor(self.delta, device=pred.device))
        lin = err - quad
        loss = 0.5 * quad.pow(2) + self.delta * lin
        loss = loss * mask
        return loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)


class CombinedDeterministicLoss(nn.Module):
    def __init__(self, mae_weight: float = 0.5, huber_delta: float = 1.0):
        super().__init__()
        self.mae_weight = mae_weight
        self.mae = MaskedMAELoss()
        self.huber = MaskedHuberLoss(delta=huber_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mae_weight * self.mae(pred, target, mask) + (1.0 - self.mae_weight) * self.huber(pred, target, mask)


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
    return CombinedDeterministicLoss(
        mae_weight=config["loss"].get("deterministic_mae_weight", 0.5),
        huber_delta=config["loss"].get("deterministic_huber_delta", 1.0),
    )


def is_probabilistic_head(head_name: str) -> bool:
    return head_name.endswith("_gauss")
