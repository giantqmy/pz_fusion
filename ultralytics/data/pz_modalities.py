"""Paired PZ detection input: R, G, B, DoLP, raw uint16 depth."""

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils.patches import imread

DEPTH_MEAN_M = 29.652361724372745
DEPTH_STD_M = 23.830449242515485


def load_rgb_dolp(path: str | Path) -> np.ndarray:
    """Read paired RGB and grayscale DoLP as a four-channel RGB-first uint8 array."""
    path = Path(path)
    rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
    dolp_path = path.with_name(f"{path.stem}_dolp_rgb.png")
    rgb = imread(str(rgb_path), cv2.IMREAD_COLOR)
    dolp = imread(str(dolp_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or dolp is None:
        raise FileNotFoundError(f"Missing RGB or DoLP input for {path}: {rgb_path}, {dolp_path}")
    if dolp.ndim == 3 and dolp.shape[-1] == 1:
        dolp = dolp[..., 0]
    if dolp.ndim != 2 or rgb.shape[:2] != dolp.shape:
        raise ValueError(f"Paired RGB/DoLP dimensions disagree for {path}: {rgb.shape}, {dolp.shape}")
    return np.concatenate((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), dolp[..., None]), axis=2)


def load_rgb_dolp_depth(path: str | Path) -> np.ndarray:
    """Read paired RGB, DoLP and raw uint16 depth as a five-channel float32 array."""
    path = Path(path)
    visual = load_rgb_dolp(path)
    depth_path = path.parent.parent / "depth" / f"{path.stem}_depth_dense_u16.png"
    depth = imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Missing depth input for {path}: {depth_path}")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected single-channel uint16 depth at {depth_path}, got {depth.dtype}, {depth.shape}")
    if visual.shape[:2] != depth.shape:
        raise ValueError(f"Paired image dimensions disagree for {path}")
    image = np.empty((*depth.shape, 5), dtype=np.float32)
    image[..., :4] = visual
    image[..., 4] = depth
    return image


def normalize_modalities(images: torch.Tensor) -> torch.Tensor:
    """Scale RGB/DoLP to [0, 1], depth meters to z-score; preserve missing depth as zero."""
    if images.ndim != 4 or images.shape[1] != 5:
        raise ValueError(f"Expected BCHW five-channel input, got {tuple(images.shape)}")
    images = images.float()
    depth = images[:, 4:5]
    normalized_depth = torch.where(depth > 0, (depth / 256.0 - DEPTH_MEAN_M) / DEPTH_STD_M, 0.0)
    return torch.cat((images[:, :4] / 255.0, normalized_depth), dim=1)
