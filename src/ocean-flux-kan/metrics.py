from __future__ import annotations

import torch


def masked_mae_per_horizon(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.float().unsqueeze(1)
    vals = []
    for i in range(pred.shape[1]):
        diff = torch.abs(pred[:, i : i + 1] - target[:, i : i + 1]) * mask
        vals.append((diff.sum() / (mask.sum() + 1e-8)).item())
    return vals


def masked_rmse_per_horizon(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.float().unsqueeze(1)
    vals = []
    for i in range(pred.shape[1]):
        diff = ((pred[:, i : i + 1] - target[:, i : i + 1]) ** 2) * mask
        mse = diff.sum() / (mask.sum() + 1e-8)
        vals.append(torch.sqrt(mse).item())
    return vals


def gaussian_coverage_per_horizon(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, z: float = 1.645):
    mask = mask.float().unsqueeze(1)
    vals = []
    for i in range(mu.shape[1]):
        lo = mu[:, i : i + 1] - z * sigma[:, i : i + 1]
        hi = mu[:, i : i + 1] + z * sigma[:, i : i + 1]
        covered = ((target[:, i : i + 1] >= lo) & (target[:, i : i + 1] <= hi)).float() * mask
        vals.append((covered.sum() / (mask.sum() + 1e-8)).item())
    return vals


class MetricTracker:
    def __init__(self, horizons=(3, 7, 14), probabilistic: bool = False):
        self.horizons = list(horizons)
        self.probabilistic = probabilistic
        self.reset()

    def reset(self):
        self.mae_sums = [0.0 for _ in self.horizons]
        self.rmse_sums = [0.0 for _ in self.horizons]
        self.coverage_sums = [0.0 for _ in self.horizons]
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sigma: torch.Tensor | None = None):
        mae_vals = masked_mae_per_horizon(pred, target, mask)
        rmse_vals = masked_rmse_per_horizon(pred, target, mask)
        for i in range(len(self.horizons)):
            self.mae_sums[i] += mae_vals[i]
            self.rmse_sums[i] += rmse_vals[i]
        if self.probabilistic and sigma is not None:
            cov_vals = gaussian_coverage_per_horizon(pred, sigma, target, mask)
            for i in range(len(self.horizons)):
                self.coverage_sums[i] += cov_vals[i]
        self.count += 1

    def compute(self):
        out = {}
        for i, h in enumerate(self.horizons):
            out[f"mae_{h}"] = self.mae_sums[i] / max(self.count, 1)
            out[f"rmse_{h}"] = self.rmse_sums[i] / max(self.count, 1)
            if self.probabilistic:
                out[f"cov90_{h}"] = self.coverage_sums[i] / max(self.count, 1)
        return out
