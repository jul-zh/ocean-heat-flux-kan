from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import SpatialTemporalBackbone
from .heads import build_head


class HeatFluxForecastModel(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_horizons: int, head_name: str, head_params: dict):
        super().__init__()
        self.backbone = SpatialTemporalBackbone(in_channels=in_channels, hidden_dim=hidden_dim)
        self.head = build_head(head_name=head_name, in_channels=32, out_channels=out_horizons, head_params=head_params)
        self.head_name = head_name

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            feat, attn = self.backbone(x, return_attention=True)
            out = self.head(feat)
            out["attention"] = attn
            return out
        feat = self.backbone(x, return_attention=False)
        return self.head(feat)
