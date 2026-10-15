"""Paired PZ detection input: RGB, grayscale or color DoLP, raw uint16 depth."""

from pathlib import Path

import cv2
import numpy as np
import torch

DEPTH_MEAN_M = 29.652361724372745
DEPTH_STD_M = 23.830449242515485


def load_rgb_dolp_depth(path: str | Path, dolp_root: str | Path | None = None, dolp_channels: int = 1) -> np.ndarray:
    """Load aligned modalities with one or three DoLP channels from an optional separate folder."""
    if dolp_channels not in {1, 3}:
        raise ValueError(f"Expected one or three DoLP channels, got {dolp_channels}")
    path = Path(path)
    rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
    dolp_dir = Path(dolp_root) if dolp_root is not None else path.parent
    dolp_suffix = "dolp_color_rgb" if dolp_channels == 3 else "dolp_rgb"
    dolp_path = dolp_dir / f"{path.stem}_{dolp_suffix}.png"
    depth_path = path.parent.parent / "depth" / f"{path.stem}_depth_dense_u16.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    dolp = cv2.imread(str(dolp_path), cv2.IMREAD_COLOR if dolp_channels == 3 else cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or dolp is None or depth is None:
        raise FileNotFoundError(f"Missing RGB, DoLP or depth input for {path}: {rgb_path}, {dolp_path}, {depth_path}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected single-channel uint16 depth at {depth_path}, got {depth.dtype}, {depth.shape}")
    if rgb.shape[:2] != dolp.shape[:2] or dolp.shape[:2] != depth.shape:
        raise ValueError(f"Paired image dimensions disagree for {path}")
    image = np.empty((*depth.shape, 4 + dolp_channels), dtype=np.float32)
    image[..., :3] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    image[..., 3:-1] = cv2.cvtColor(dolp, cv2.COLOR_BGR2RGB) if dolp_channels == 3 else dolp[..., None]
    image[..., -1] = depth
    return image


def normalize_modalities(images: torch.Tensor) -> torch.Tensor:
    """Scale RGB/DoLP to [0, 1], depth meters to z-score; preserve missing depth as zero."""
    if images.ndim != 4 or images.shape[1] not in {5, 7}:
        raise ValueError(f"Expected BCHW five- or seven-channel input, got {tuple(images.shape)}")
    images = images.float()
    depth = images[:, -1:]
    normalized_depth = torch.where(depth > 0, (depth / 256.0 - DEPTH_MEAN_M) / DEPTH_STD_M, 0.0)
    return torch.cat((images[:, :-1] / 255.0, normalized_depth), dim=1)
