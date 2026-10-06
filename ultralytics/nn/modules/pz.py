"""Small PIM/SAFF-style blocks for the five-channel PZ YOLO26 model."""

import torch
import torch.nn.functional as F
from torch import nn


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.pw = nn.Sequential(nn.Conv2d(channels, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1))

    def forward(self, x):
        return x + self.pw(self.norm(self.dw(x)))


class PZPIMStage(nn.Module):
    """Visual RGB+DoLP stage; the first stage consumes the first four input channels."""

    def __init__(self, c_in, c_out, stride=2, depth=1):
        super().__init__()
        self.first = c_in == 5
        visual_in = 4 if self.first else c_in
        self.proj = nn.Sequential(
            nn.Conv2d(visual_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
            *(_ConvNeXtBlock(c_out) for _ in range(max(1, depth))),
        )
        self.pim = nn.Sequential(nn.Conv2d(c_out, c_out, 1, groups=1, bias=False), nn.BatchNorm2d(c_out), nn.SiLU())

    def forward(self, x):
        if self.first:
            x = x[:, :4]
        return self.pim(self.proj(x))


class PZDepthStage(nn.Module):
    """One cascaded depth stage. The first stage extracts channel 4 from raw five-channel input."""

    def __init__(self, c_in, stride=2, width=32):
        super().__init__()
        self.from_raw = c_in == 5
        self.block = nn.Sequential(
            nn.Conv2d(1 if self.from_raw else c_in, width, 3, stride=int(stride), padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            _ConvNeXtBlock(width),
        )

    def forward(self, x):
        if self.from_raw:
            x = x[:, 4:5]
        return self.block(x)


class PZDepthFusion(nn.Module):
    """Fuse the already-cascaded depth feature into a visual feature with a small residual gate."""

    def __init__(self, channels, depth_channels=32):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(self, inputs):
        visual, depth = inputs
        depth = self.depth_proj(depth)
        if depth.shape[-2:] != visual.shape[-2:]:
            depth = F.interpolate(depth, size=visual.shape[-2:], mode="bilinear", align_corners=False)
        return visual + torch.sigmoid(self.gate(torch.cat((visual, depth), dim=1))) * depth
