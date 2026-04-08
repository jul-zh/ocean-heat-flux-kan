from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, hidden_dim)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        return x3, x1


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 2, 1)
        self.score = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        scores = []
        for t in range(x.shape[1]):
            scores.append(self.score(x[:, t]))
        scores = torch.stack(scores, dim=1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights


class SharedDecoderBody(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(hidden_dim + 16, 32)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [0, diff_w, 0, diff_h])
        x = torch.cat([x, skip], dim=1)
        return self.dec1(x)


class SpatialTemporalBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = SmallEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.temporal_pool = TemporalAttentionPool(channels=hidden_dim)
        self.decoder_body = SharedDecoderBody(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feats = []
        last_skip = None
        for step in range(x.shape[1]):
            feat, skip = self.encoder(x[:, step])
            feats.append(feat)
            last_skip = skip
        feats = torch.stack(feats, dim=1)
        pooled, attn = self.temporal_pool(feats)
        dec_feat = self.decoder_body(pooled, last_skip)
        if return_attention:
            return dec_feat, attn
        return dec_feat
