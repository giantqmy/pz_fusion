"""PZ visual/depth blocks, including dual-branch PIM/FAM and sparse SAFF fusion."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class FusePol(nn.Module):
    """Fuse raw three-channel AoLP and DoLP, preserving RGB as a separate branch."""

    def __init__(self, c1: int, c2: int = 6):
        super().__init__()
        if c1 != 9 or c2 != 6:
            raise ValueError(f"FusePol expects 9 input and 6 output channels, got {c1} and {c2}")
        self.conv_sattn = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
        self.degree_max_map = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, bias=False), nn.BatchNorm2d(3), nn.SiLU(),
            nn.MaxPool2d(5, stride=1, padding=2), nn.Sigmoid(),
        )
        self.conv_aop = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1, bias=False), nn.BatchNorm2d(3), nn.SiLU())
        self.conv_dop = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1, bias=False), nn.BatchNorm2d(3), nn.SiLU())
        self.conv_pol = nn.Sequential(nn.Conv2d(6, 3, 3, padding=1, bias=False), nn.BatchNorm2d(3), nn.SiLU())
        self.register_buffer("scharr_x", torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)[None, None])
        self.register_buffer("scharr_y", torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)[None, None])

    def forward(self, x):
        rgb, aop, dop = x[:, :3], x[:, 3:6], x[:, 6:9]
        maximum, _ = dop.max(dim=1, keepdim=True)
        average = dop.mean(dim=1, keepdim=True)
        weighted_aop = aop * (self.conv_sattn(torch.cat((maximum, average), dim=1)) + self.degree_max_map(dop))
        gray_dop = dop.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray_dop, self.scharr_x.to(dtype=x.dtype), padding=1)
        grad_y = F.conv2d(gray_dop, self.scharr_y.to(dtype=x.dtype), padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        weighted_dop = dop + edge
        pol = self.conv_pol(torch.cat((self.conv_aop(weighted_aop), self.conv_dop(weighted_dop)), dim=1))
        return torch.cat((rgb, pol), dim=1)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.pw = nn.Sequential(nn.Conv2d(channels, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1))

    def forward(self, x):
        return x + self.pw(self.norm(self.dw(x)))


class PZPIMStage(nn.Module):
    """Visual RGB+DoLP stage; the first stage excludes the final depth channel."""

    def __init__(self, c_in, c_out, stride=2, depth=1):
        super().__init__()
        self.first = c_in in {5, 7}
        visual_in = c_in - 1 if self.first else c_in
        self.proj = nn.Sequential(
            nn.Conv2d(visual_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
            *(_ConvNeXtBlock(c_out) for _ in range(max(1, depth))),
        )
        self.pim = nn.Sequential(nn.Conv2d(c_out, c_out, 1, groups=1, bias=False), nn.BatchNorm2d(c_out), nn.SiLU())

    def forward(self, x):
        if self.first:
            x = x[:, :-1]
        return self.pim(self.proj(x))


class _VisualConvNeXtBlock(nn.Module):
    """ConvNeXt with LayerNorm and layer scale, matching the original dual visual backbone."""

    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))
        nn.init.normal_(self.dwconv.weight, std=math.sqrt(2.0 / 49))
        nn.init.zeros_(self.dwconv.bias)
        for layer in (self.pwconv1, self.pwconv2):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        y = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + y.permute(0, 3, 1, 2)


class _ChannelLayerNorm(nn.LayerNorm):
    """Apply LayerNorm over channels of an NCHW feature tensor."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _HaarDWT(nn.Module):
    """Fixed per-channel Haar decomposition with padding for odd feature sizes."""

    def __init__(self, channels):
        super().__init__()
        filters = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, 1.0]],
             [[-1.0, 1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
        ) / 2
        self.register_buffer("weight", filters[:, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = F.pad(x, (0, width % 2, 0, height % 2))
        bands = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=channels)
        bands = bands.reshape(batch, channels, 4, *bands.shape[-2:])
        return bands[:, :, 0], torch.cat(tuple(bands[:, :, i] for i in (1, 2, 3)), dim=1)


class _WaveletS0DoLPFusion(nn.Module):
    """Fuse RGB intensity and DoLP independently in low and high Haar bands."""

    def __init__(self, channels):
        super().__init__()
        self.dwt = _HaarDWT(channels)

        def gate(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False), nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True), nn.Conv2d(channels, 2, 1),
            )

        def proj(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True)
            )

        self.low_gate = gate(channels * 2)
        self.high_gate = gate(channels * 6)
        self.high_proj = proj(channels * 3)
        self.out_proj = proj(channels * 2)

    def forward(self, rgb, dolp):
        low_rgb, high_rgb = self.dwt(rgb)
        low_dolp, high_dolp = self.dwt(dolp)
        low_w = self.low_gate(torch.cat((low_rgb, low_dolp), dim=1)).softmax(dim=1)
        high_w = self.high_gate(torch.cat((high_rgb, high_dolp), dim=1)).softmax(dim=1)
        low = low_rgb * low_w[:, :1] + low_dolp * low_w[:, 1:]
        high = self.high_proj(high_rgb * high_w[:, :1] + high_dolp * high_w[:, 1:])
        low = F.interpolate(low, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        high = F.interpolate(high, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_proj(torch.cat((low, high), dim=1))


class _WaveletPIM(nn.Module):
    """PoolFormer branch refinement and shared wavelet residual from the original PIM."""

    def __init__(self, channels):
        super().__init__()
        self.norm1_s0 = nn.BatchNorm2d(channels)
        self.norm1_dolp = nn.BatchNorm2d(channels)
        self.norm2_s0 = nn.BatchNorm2d(channels)
        self.norm2_dolp = nn.BatchNorm2d(channels)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.mlp_s0 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1)
        )
        self.mlp_dolp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1)
        )
        self.wavelet_fusion = _WaveletS0DoLPFusion(channels)
        self.layer_scale_1_s0 = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_1_dolp = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_2_s0 = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_2_dolp = nn.Parameter(torch.full((channels,), 1e-5))
        self.fusion_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _scale(gamma, x):
        return gamma.view(1, -1, 1, 1) * x

    def forward(self, rgb, dolp):
        n_rgb, n_dolp = self.norm1_s0(rgb), self.norm1_dolp(dolp)
        rgb = rgb + self._scale(self.layer_scale_1_s0, self.pool(n_rgb) - n_rgb)
        dolp = dolp + self._scale(self.layer_scale_1_dolp, self.pool(n_dolp) - n_dolp)
        n_rgb, n_dolp = self.norm2_s0(rgb), self.norm2_dolp(dolp)
        fusion = self.fusion_scale * self.wavelet_fusion(n_rgb, n_dolp)
        rgb = rgb + self._scale(self.layer_scale_2_s0, self.mlp_s0(n_rgb))
        dolp = dolp + self._scale(self.layer_scale_2_dolp, self.mlp_dolp(n_dolp))
        return rgb + fusion, dolp + fusion


class _HaarIDWT(nn.Module):
    """Fixed inverse of ``_HaarDWT`` for four C-channel Haar subbands."""

    def __init__(self, channels):
        super().__init__()
        self.dwt = _HaarDWT(channels)
        self.channels = channels

    def forward(self, ll, lh, hl, hh, output_size=None):
        coeffs = torch.stack((ll, lh, hl, hh), dim=2).flatten(1, 2)
        x = F.conv_transpose2d(coeffs, self.dwt.weight.to(dtype=coeffs.dtype), stride=2, groups=self.channels)
        if output_size is not None:
            x = x[..., : output_size[0], : output_size[1]]
        return x


class pim_dualWT(nn.Module):
    """Dual RGB/DoLP wavelet fusion with difference-guided per-band gates.

    Each modality is decomposed into LL/LH/HL/HH bands. Every band has an
    independent gate that uses mapped-modal differences as auxiliary context,
    then the four fused C-channel bands are reconstructed by fixed Haar IDWT.
    """

    _BANDS = ("ll", "lh", "hl", "hh")

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("pim_dualWT requires a positive channel count")
        self.channels = channels
        self.dwt = _HaarDWT(channels)
        self.idwt = _HaarIDWT(channels)
        self.rgb_compare = nn.ModuleList(nn.Conv2d(channels, channels, 1, bias=False) for _ in self._BANDS)
        self.dolp_compare = nn.ModuleList(nn.Conv2d(channels, channels, 1, bias=False) for _ in self._BANDS)
        hidden = max(4, channels // 4)
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels * 3, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 2, 1),
            )
            for _ in self._BANDS
        )
        self.adapter_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.adapter_dolp = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma_rgb = nn.Parameter(torch.full((channels,), 1e-3))
        self.gamma_dolp = nn.Parameter(torch.full((channels,), 1e-3))

    def _decompose(self, x):
        ll, high = self.dwt(x)
        return (ll, *high.chunk(3, dim=1))

    def _fuse_band(self, rgb, dolp, index):
        rgb_mapped = self.rgb_compare[index](rgb)
        dolp_mapped = self.dolp_compare[index](dolp)
        difference = (rgb_mapped - dolp_mapped).abs()
        logits = self.gates[index](torch.cat((rgb, dolp, difference), dim=1))
        weights = logits.softmax(dim=1)
        fused = rgb * weights[:, :1] + dolp * weights[:, 1:2]
        return fused, weights

    def forward(self, rgb, dolp, return_weights=False):
        if rgb.shape != dolp.shape or rgb.ndim != 4:
            raise ValueError("pim_dualWT expects RGB and DoLP tensors with the same BCHW shape")
        height, width = rgb.shape[-2:]
        rgb_bands = self._decompose(rgb)
        dolp_bands = self._decompose(dolp)
        fused_bands, weights = [], []
        for index, (rgb_band, dolp_band) in enumerate(zip(rgb_bands, dolp_bands)):
            fused, band_weights = self._fuse_band(rgb_band, dolp_band, index)
            fused_bands.append(fused)
            weights.append(band_weights)
        fused = self.idwt(*fused_bands, output_size=(height, width))
        rgb_out = rgb + self.gamma_rgb.view(1, -1, 1, 1) * self.adapter_rgb(fused)
        dolp_out = dolp + self.gamma_dolp.view(1, -1, 1, 1) * self.adapter_dolp(fused)
        if return_weights:
            return (rgb_out, dolp_out), tuple(weights)
        return rgb_out, dolp_out


class _DualVisualStage(nn.Module):
    """Shared RGB/DoLP stems and ConvNeXt blocks for PIM and FAM stages.

    Args:
        c_in (int): Five, six or seven raw input channels or the preceding dual state's channel count.
        c_out (int): Total output channels, split equally between the two branches.
        stride (int): Downsample stride, either 2 or a two-step stride 4 stem.
        depth (int): Number of ConvNeXt blocks per branch.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 2, depth: int = 1):
        super().__init__()
        if c_out % 2 or stride not in (2, 4) or depth < 1:
            raise ValueError("Dual visual stages require even output channels, stride 2/4 and positive depth")
        self.first = c_in in {5, 7}
        self.branch_channels = (3, c_in - 4) if self.first else (c_in // 2, c_in // 2)
        if not self.first and c_in % 2:
            raise ValueError("Dual visual backbone states must have even channels")
        self.branch1 = self._make_branch(self.branch_channels[0], c_out // 2, stride, depth)
        self.branch2 = self._make_branch(self.branch_channels[1], c_out // 2, stride, depth)

    def _make_branch(self, in_channels, half, stride, depth):
        if stride == 4:
            mid = max(1, half // 2)
            stem = [
                nn.Conv2d(in_channels, mid, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(mid), nn.SiLU(),
                nn.Conv2d(mid, half, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(half), nn.SiLU(),
            ]
        else:
            stem = [nn.Conv2d(in_channels, half, 2, stride=2, bias=False), nn.BatchNorm2d(half), nn.SiLU()]
        return nn.Sequential(*stem, *(_VisualConvNeXtBlock(half) for _ in range(depth)))

    def _features(self, x):
        if self.first:
            x = x[:, :-1]
        rgb, dolp = torch.split(x, self.branch_channels, dim=1)
        return self.branch1(rgb), self.branch2(dolp)


class PZDualPIMStage(_DualVisualStage):
    """Preserve dual ConvNeXt states with wavelet PIM interaction.

    Args:
        c_in (int): Five, six or seven raw input channels or the preceding dual state's channel count.
        c_out (int): Total output channels, split equally between the two branches.
        stride (int): Downsample stride, either 2 or a two-step stride 4 stem.
        depth (int): Number of ConvNeXt blocks per branch.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 2, depth: int = 1):
        super().__init__(c_in, c_out, stride, depth)
        self.pim = _WaveletPIM(c_out // 2)

    def forward(self, x):
        rgb, dolp = self.pim(*self._features(x))
        return torch.cat((rgb, dolp), dim=1)


class PZDualWTStage(_DualVisualStage):
    """Dual ConvNeXt visual stage using ``pim_dualWT`` instead of legacy PIM."""

    def __init__(self, c_in: int, c_out: int, stride: int = 2, depth: int = 1):
        super().__init__(c_in, c_out, stride, depth)
        self.pim = pim_dualWT(c_out // 2)

    def forward(self, x):
        rgb, dolp = self.pim(*self._features(x))
        return torch.cat((rgb, dolp), dim=1)


class _CBS(nn.Sequential):
    """Conv-BN-SiLU block used by the edge-aware wavelet PIM."""

    def __init__(self, c1, c2):
        super().__init__(nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))


class _SobelLaplaceEdge(nn.Module):
    """Fixed per-channel Sobel and Laplacian responses followed by a CBS mixer."""

    def __init__(self, channels):
        super().__init__()
        kernels = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        high_channels = channels * 3
        self.register_buffer("kernels", kernels[:, None].repeat(high_channels, 1, 1, 1))
        self.channels = high_channels
        self.mix = _CBS(high_channels * 3, high_channels)

    def forward(self, x):
        responses = F.conv2d(x, self.kernels.to(dtype=x.dtype), padding=1, groups=self.channels)
        return self.mix(responses.abs())


class _DualWaveletDifferenceGate(nn.Module):
    """Difference-guided CBAM residual gate shared across the two modalities."""

    def __init__(self, channels):
        super().__init__()
        from .conv import CBAM

        self.cbam = CBAM(channels)

    def forward(self, rgb, dolp):
        # CBAM already returns a sigmoid-bounded attention map.
        gate = self.cbam((rgb - dolp).abs())
        return rgb + rgb * gate, dolp + dolp * gate


class _WaveletEdgeCBAMPIM(nn.Module):
    """Band-wise modality-difference fusion with a DoLP high-frequency edge path."""

    def __init__(self, channels, edge_init=1.0):
        super().__init__()
        self.channels = channels
        self.dwt = _HaarDWT(channels)
        self.idwt = _HaarIDWT(channels)
        self.rgb_high = _CBS(channels * 3, channels * 3)
        self.dolp_high = _CBS(channels * 3, channels * 3)
        self.dolp_edge = _SobelLaplaceEdge(channels)
        self.edge_gain = nn.Parameter(torch.tensor(float(edge_init)))
        self.rgb_low = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.dolp_low = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.high_gate = _DualWaveletDifferenceGate(channels * 3)
        self.low_gate = _DualWaveletDifferenceGate(channels)

    def _bands(self, x):
        low, high = self.dwt(x)
        return low, high

    def forward(self, rgb, dolp):
        size = rgb.shape[-2:]
        rgb_low, rgb_high = self._bands(rgb)
        dolp_low, dolp_high = self._bands(dolp)

        rgb_high = self.rgb_high(rgb_high)
        dolp_high = self.dolp_high(dolp_high)
        dolp_high = dolp_high + self.edge_gain * self.dolp_edge(dolp_high)
        rgb_low = self.rgb_low(rgb_low)
        dolp_low = self.dolp_low(dolp_low)

        rgb_high, dolp_high = self.high_gate(rgb_high, dolp_high)
        rgb_low, dolp_low = self.low_gate(rgb_low, dolp_low)

        rgb_out = self.idwt(rgb_low, *rgb_high.chunk(3, dim=1), output_size=size)
        dolp_out = self.idwt(dolp_low, *dolp_high.chunk(3, dim=1), output_size=size)
        return rgb_out, dolp_out


class PZDualEdgeWTStage(_DualVisualStage):
    """Dual ConvNeXt stage with Sobel/Laplacian and CBAM-gated wavelet fusion."""

    def __init__(self, c_in: int, c_out: int, stride: int = 2, depth: int = 1):
        super().__init__(c_in, c_out, stride, depth)
        self.pim = _WaveletEdgeCBAMPIM(c_out // 2)

    def forward(self, x):
        rgb, dolp = self.pim(*self._features(x))
        return torch.cat((rgb, dolp), dim=1)


class PZDualEdgeWTZeroStage(_DualVisualStage):
    """Edge-CBAM wavelet stage with a zero-initialized learnable edge residual."""

    def __init__(self, c_in: int, c_out: int, stride: int = 2, depth: int = 1, edge_init: float = 0.0):
        super().__init__(c_in, c_out, stride, depth)
        self.pim = _WaveletEdgeCBAMPIM(c_out // 2, edge_init=edge_init)

    def forward(self, x):
        rgb, dolp = self.pim(*self._features(x))
        return torch.cat((rgb, dolp), dim=1)


class PZConvNeXtPIMStage(PZDualPIMStage):
    """Extract dual visual features with ConvNeXt stems/downsampling before wavelet PIM.

    Args:
        c_in (int): Five raw input channels or the preceding dual state's channel count.
        c_out (int): Total output channels, split equally between the two branches.
        stride (int): Downsample stride, 4 for the patch stem or 2 for later stages.
        depth (int): Number of ConvNeXt blocks per branch.
    """

    def _make_branch(self, in_channels, half, stride, depth):
        if stride == 4:
            stem = [nn.Conv2d(in_channels, half, 4, stride=4), _ChannelLayerNorm(half, eps=1e-6)]
        else:
            stem = [_ChannelLayerNorm(in_channels, eps=1e-6), nn.Conv2d(in_channels, half, 2, stride=2)]
        return nn.Sequential(*stem, *(_VisualConvNeXtBlock(half) for _ in range(depth)))


class _FAMChannelWeights(nn.Module):
    """Original FAM channel gates from modality-weighted average and maximum descriptors."""

    def __init__(self, dim: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_avg = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2))
        self.mlp_max = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, rgb, dolp):
        batch, channels = rgb.shape[:2]
        x = torch.cat((rgb, dolp), dim=1)
        avg = self.avg_pool(x).view(batch, 2 * channels)
        avg_attn = self.mlp_avg(avg).softmax(dim=-1)
        avg_rgb, avg_dolp = (avg_attn.view(batch, 2, 1) * avg.view(batch, 2, channels)).chunk(2, dim=1)
        avg_x = (avg_rgb + avg_dolp).view(batch, channels)
        maximum = self.max_pool(x).view(batch, 2 * channels)
        max_attn = self.mlp_max(maximum).softmax(dim=-1)
        max_rgb, max_dolp = (max_attn.view(batch, 2, 1) * maximum.view(batch, 2, channels)).chunk(2, dim=1)
        max_x = (max_rgb + max_dolp).view(batch, channels)
        weights = self.mlp(torch.cat((avg_x, max_x), dim=1))
        return weights.reshape(batch, 2, channels, 1, 1).permute(1, 0, 2, 3, 4)


class _FAMSpatialWeights(nn.Module):
    """Original FAM spatial gates from concatenated RGB and DoLP features."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.ReLU(inplace=True), nn.Conv2d(dim, 2, 1), nn.Sigmoid())

    def forward(self, rgb, dolp):
        batch, _, height, width = rgb.shape
        weights = self.mlp(torch.cat((rgb, dolp), dim=1))
        return weights.reshape(batch, 2, 1, height, width).permute(1, 0, 2, 3, 4)


class _FeatureAlignmentModule(nn.Module):
    """Match PZ_8 FAM with reduction=1 and fixed channel/spatial residual scales of 0.5."""

    def __init__(self, dim: int):
        super().__init__()
        self.lambda_c = 0.5
        self.lambda_s = 0.5
        self.channel_weights = _FAMChannelWeights(dim)
        self.spatial_weights = _FAMSpatialWeights(dim)

    def forward(self, rgb, dolp):
        channel = self.channel_weights(rgb, dolp)
        spatial = self.spatial_weights(rgb, dolp)
        out_rgb = rgb + self.lambda_c * channel[1] * dolp + self.lambda_s * spatial[1] * dolp
        out_dolp = dolp + self.lambda_c * channel[0] * rgb + self.lambda_s * spatial[0] * rgb
        return out_rgb, out_dolp


class PZDualFAMStage(_DualVisualStage):
    """Preserve dual ConvNeXt states with the channel/spatial FAM from PZ_8.

    Args:
        c_in (int): Five raw input channels or the preceding dual state's channel count.
        c_out (int): Total output channels, split equally between the two branches.
        stride (int): Downsample stride, either 2 or a two-step stride 4 stem.
        depth (int): Number of ConvNeXt blocks per branch.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 2, depth: int = 1):
        super().__init__(c_in, c_out, stride, depth)
        self.fam = _FeatureAlignmentModule(c_out)

    def forward(self, x):
        rgb, dolp = self.fam(*self._features(x))
        return torch.cat((rgb, dolp), dim=1)


class _SparseSAFF(nn.Module):
    """Top-k window attention with aligned K1/V2 and K2/V1 routing, matching original SAFF."""

    def __init__(self, channels, topk, n_win, kv_ratio):
        super().__init__()
        if not 0 < topk <= n_win * n_win or kv_ratio < 1:
            raise ValueError("SAFF requires positive KV ratio and topk within the window count")
        self.channels, self.topk, self.n_win = channels, topk, n_win
        self.kv_ratio = kv_ratio
        self.scale = channels**-0.5
        self.qkv1 = nn.Linear(channels, 3 * channels)
        self.qkv2 = nn.Linear(channels, 3 * channels)
        self.kv_down = nn.AvgPool2d(kv_ratio) if kv_ratio > 1 else nn.Identity()
        self.residual1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.residual2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def _partition(self, x):
        batch, channels, height, width = x.shape
        return x.reshape(batch, channels, self.n_win, height // self.n_win, self.n_win, width // self.n_win).permute(
            0, 2, 4, 3, 5, 1
        ).reshape(batch, self.n_win**2, height // self.n_win, width // self.n_win, channels)

    def _flatten_kv(self, kv):
        batch, windows, height, width, channels = kv.shape
        kv = self.kv_down(kv.permute(0, 1, 4, 2, 3).reshape(batch * windows, channels, height, width))
        return kv.reshape(batch, windows, channels, -1).transpose(-2, -1)

    def _attention_values(self, kv1, kv2):
        return kv2[..., self.channels:], kv1[..., self.channels:]

    def _attend(self, query, own_kv, value):
        channels = self.channels
        q_win = query.mean((2, 3)).detach()
        k_win = own_kv[..., :channels].mean((2, 3)).detach()
        route = ((q_win * self.scale) @ k_win.transpose(-2, -1)).topk(self.topk, dim=-1).indices
        key, value = self._flatten_kv(own_kv[..., :channels]), self._flatten_kv(value)
        kv = torch.cat((key, value), dim=-1)
        batch, windows, pixels, features = kv.shape
        selected = torch.gather(
            kv[:, None].expand(-1, windows, -1, -1, -1), 2,
            route[..., None, None].expand(-1, -1, -1, pixels, features),
        ).reshape(batch, windows, self.topk * pixels, features)
        key, value = selected.split(channels, dim=-1)
        query = query.flatten(2, 3)
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        return attention @ value

    def _restore(self, x, height, width):
        return (
            x.reshape(x.shape[0], self.n_win, self.n_win, height // self.n_win, width // self.n_win, self.channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(x.shape[0], self.channels, height, width)
        )

    def forward(self, rgb, dolp):
        if rgb.shape != dolp.shape:
            raise ValueError("SAFF requires aligned RGB and DoLP features")
        height, width = rgb.shape[-2:]
        residual_rgb, residual_dolp = self.residual1(rgb), self.residual2(dolp)
        # Rectangular validation inputs can otherwise produce windows smaller than the K/V pooling kernel.
        padded_h = max(math.ceil(height / self.n_win), self.kv_ratio) * self.n_win
        padded_w = max(math.ceil(width / self.n_win), self.kv_ratio) * self.n_win
        pad = (0, padded_w - width, 0, padded_h - height)
        rgb, dolp = F.pad(rgb, pad), F.pad(dolp, pad)
        q1, kv1 = self.qkv1(self._partition(rgb)).split((self.channels, 2 * self.channels), dim=-1)
        q2, kv2 = self.qkv2(self._partition(dolp)).split((self.channels, 2 * self.channels), dim=-1)
        value1, value2 = self._attention_values(kv1, kv2)
        # Each branch routes its own keys and gathers values at the same window indices.
        to_dolp = self._restore(self._attend(q1, kv1, value1), *rgb.shape[-2:])[..., :height, :width]
        to_rgb = self._restore(self._attend(q2, kv2, value2), *rgb.shape[-2:])[..., :height, :width]
        return residual_rgb + to_rgb, residual_dolp + to_dolp


class _SparseSAFFConcat(_SparseSAFF):
    """Route each modality's own Q/K over a shared linear fusion of aligned RGB/DoLP values."""

    def __init__(self, channels: int, topk: int, n_win: int, kv_ratio: int):
        super().__init__(channels, topk, n_win, kv_ratio)
        self.value_fuse = nn.Linear(2 * channels, channels)

    def _attention_values(self, kv1, kv2):
        fused = self.value_fuse(torch.cat((kv1[..., self.channels:], kv2[..., self.channels:]), dim=-1))
        return fused, fused


class PZSAFFLateral(nn.Module):
    """Fuse a dual PIM state for the PAN lateral while leaving the backbone state separate.

    Args:
        channels (int): Total channels of the concatenated RGB and DoLP state.
        topk (int): Number of routed windows per query window.
        n_win (int): Number of windows along each spatial axis.
        kv_ratio (int): Average pooling ratio within each K/V window.
    """

    attention_type = _SparseSAFF

    def __init__(self, channels: int, topk: int = 4, n_win: int = 8, kv_ratio: int = 4):
        super().__init__()
        if channels % 2:
            raise ValueError("SAFF lateral requires even channels")
        self.saff = self.attention_type(channels // 2, topk, n_win, kv_ratio)
        self.out_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())

    def forward(self, x):
        rgb, dolp = self.saff(*x.chunk(2, dim=1))
        return self.out_fuse(torch.cat((rgb, dolp), dim=1))


class PZSAFFConcatLateral(PZSAFFLateral):
    """Fuse SAFFConcat's two residual outputs into one visual lateral for depth HMoE.

    Args:
        channels (int): Total channels of the concatenated RGB and DoLP state.
        topk (int): Number of routed windows per query window.
        n_win (int): Number of windows along each spatial axis.
        kv_ratio (int): Average pooling ratio within each K/V window.
    """

    attention_type = _SparseSAFFConcat


class PZDepthStage(nn.Module):
    """One cascaded depth stage. The first stage extracts the final raw input channel."""

    def __init__(self, c_in, stride=2, width=32):
        super().__init__()
        self.from_raw = c_in in {5, 7}
        self.block = nn.Sequential(
            nn.Conv2d(1 if self.from_raw else c_in, width, 3, stride=int(stride), padding=1, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(), _ConvNeXtBlock(width),
        )

    def forward(self, x):
        # A singleton ``from`` list can be produced by model YAML parsing.
        # Depth stages consume one tensor; only fusion stages intentionally
        # receive a list of tensors.
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(f"PZDepthStage expects one tensor, got {len(x)} inputs")
            x = x[0]
        if self.from_raw:
            x = x[:, -1:]
        return self.block(x)


class PZDepthFusion(nn.Module):
    """Fuse the already-cascaded depth feature into a visual feature with a small residual gate."""

    def __init__(self, channels, depth_channels=32):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.depth_proj = nn.Sequential(nn.Conv2d(depth_channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(self, inputs):
        visual, depth = inputs
        depth = self.depth_proj(depth)
        if depth.shape[-2:] != visual.shape[-2:]:
            depth = F.interpolate(depth, size=visual.shape[-2:], mode="bilinear", align_corners=False)
        return visual + torch.sigmoid(self.gate(torch.cat((visual, depth), dim=1))) * depth


class PZHMoEFusion(nn.Module):
    """Update visual features using SEATrack-style soft routing over visual and depth tokens."""

    def __init__(self, channels, depth_channels=32, experts=4, slots=2, rank=4):
        super().__init__()
        if channels % slots:
            raise ValueError("HMoE channels must be divisible by slots")
        self.experts, self.slots = experts, slots
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )
        self.norm = nn.LayerNorm(channels)
        self.linear1 = self._projection(channels, rank)
        self.linear2 = self._projection(channels, rank)
        sub_dim = channels // slots
        self.gate_thi = nn.Parameter(torch.randn(sub_dim, experts * slots) * sub_dim**-0.5)
        self.expert_a = nn.Parameter(torch.empty(experts, sub_dim, rank))
        self.expert_b = nn.Parameter(torch.empty(experts, rank, sub_dim))
        self.bias_a = nn.Parameter(torch.zeros(experts, 1, rank))
        self.bias_b = nn.Parameter(torch.zeros(experts, 1, sub_dim))
        nn.init.xavier_normal_(self.expert_a, gain=2**0.5)
        nn.init.xavier_normal_(self.expert_b, gain=2**0.5)
        self.drop = nn.Dropout(0.1)
        # softplus(raw) + epsilon starts at 1 and keeps both temperatures positive.
        self.dispatch_temp = nn.Parameter(torch.tensor(0.54116665))
        self.combine_temp = nn.Parameter(torch.tensor(0.54116665))
        self.alpha = nn.Parameter(torch.tensor(0.01))

    @staticmethod
    def _projection(channels, rank):
        return nn.Sequential(nn.Linear(channels, rank), nn.Dropout(0.1), nn.Linear(rank, channels), nn.Dropout(0.1))

    def _readout(self, mixed, pixels):
        return mixed[:, :pixels]

    def forward(self, inputs):
        visual, depth = inputs
        depth = self.depth_proj(depth)
        if depth.shape[-2:] != visual.shape[-2:]:
            depth = F.interpolate(depth, size=visual.shape[-2:], mode="bilinear", align_corners=False)
        batch, channels, height, width = visual.shape
        pixels = height * width
        tokens = torch.cat((visual.flatten(2).transpose(1, 2), depth.flatten(2).transpose(1, 2)), dim=1)
        count = tokens.shape[1]
        x = self.linear1(self.norm(tokens)).reshape(batch, count * self.slots, channels // self.slots)
        # Normalize routing scores in float32 under mixed precision.
        logits = (x @ self.gate_thi).float()
        dispatch = (logits / (F.softplus(self.dispatch_temp.float()) + 1e-4)).softmax(dim=1)
        combine = logits.reshape(batch, count, self.slots, self.experts, self.slots).sum(dim=(2, 4))
        combine = (combine / (F.softplus(self.combine_temp.float()) + 1e-4)).softmax(dim=-1)
        expert_inputs = torch.bmm(dispatch.transpose(1, 2).to(x.dtype), x)
        expert_inputs = expert_inputs.reshape(batch, self.experts, self.slots, channels // self.slots)
        outputs = self.drop(expert_inputs @ self.expert_a + self.bias_a) @ self.expert_b + self.bias_b
        outputs = self.linear2(self.drop(outputs).reshape(batch, self.experts, channels))
        mixed = torch.bmm(combine.to(outputs.dtype), outputs)
        delta = self._readout(mixed, pixels).transpose(1, 2).reshape(batch, channels, height, width)
        return visual + self.alpha * delta


class PZHMoEJointFusion(PZHMoEFusion):
    """Project paired visual/depth HMoE outputs from 2C to C channels at each spatial position."""

    def __init__(self, channels: int, depth_channels: int = 32, experts: int = 4, slots: int = 2, rank: int = 4):
        super().__init__(channels, depth_channels, experts, slots, rank)
        self.joint_proj = nn.Linear(2 * channels, channels)

    def _readout(self, mixed, pixels):
        paired = torch.cat((mixed[:, :pixels], mixed[:, pixels:]), dim=-1)
        return self.joint_proj(paired)
