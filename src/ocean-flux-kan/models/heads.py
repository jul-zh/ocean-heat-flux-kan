from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"pred": self.out(x)}


class MLPDetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.net(y)
        y = y.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return {"pred": y}


class GaussianConvHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.out(x)
        mu, raw_sigma = out.chunk(2, dim=1)
        sigma = F.softplus(raw_sigma) + 1e-6
        return {"mu": mu, "sigma": sigma}


class RBFBasisLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_centers: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.linspace(-2.0, 2.0, num_centers).repeat(in_features, 1))
        self.log_gamma = nn.Parameter(torch.zeros(in_features, num_centers))
        self.weight = nn.Parameter(torch.randn(in_features, num_centers, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_features]
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)  # [N, F, K]
        gamma = torch.exp(self.log_gamma).unsqueeze(0)
        phi = torch.exp(-gamma * diff.pow(2))
        out = torch.einsum("nfk,fko->no", phi, self.weight) + self.bias
        return out


class RBFKANDetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rbf_bins: int = 16):
        super().__init__()
        self.kan = RBFBasisLayer(in_channels, out_channels, num_centers=rbf_bins)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.kan(y)
        y = y.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return {"pred": y}


class RBFKANGaussianHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rbf_bins: int = 16):
        super().__init__()
        self.kan = RBFBasisLayer(in_channels, 2 * out_channels, num_centers=rbf_bins)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.kan(y)
        y = y.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        mu, raw_sigma = y.chunk(2, dim=1)
        sigma = F.softplus(raw_sigma) + 1e-6
        return {"mu": mu, "sigma": sigma}


def build_head(head_name: str, in_channels: int, out_channels: int, head_params: dict):
    if head_name == "conv_det":
        return ConvDetHead(in_channels, out_channels)
    if head_name == "mlp_det":
        return MLPDetHead(in_channels, out_channels, hidden=head_params.get("kan_hidden", 64), dropout=head_params.get("dropout", 0.0))
    if head_name == "rbf_kan_det":
        return RBFKANDetHead(in_channels, out_channels, rbf_bins=head_params.get("rbf_bins", 16))
    if head_name == "conv_gauss":
        return GaussianConvHead(in_channels, out_channels)
    if head_name == "rbf_kan_gauss":
        return RBFKANGaussianHead(in_channels, out_channels, rbf_bins=head_params.get("rbf_bins", 16))
    raise ValueError(f"Unsupported head_name={head_name}")
