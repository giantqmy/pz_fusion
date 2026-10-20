"""Paired PZ detection input: RGB, grayscale or color DoLP, raw uint16 depth."""

from pathlib import Path

import cv2
import numpy as np
import torch

DEPTH_MEAN_M = 29.652361724372745
DEPTH_STD_M = 23.830449242515485


def _load_rgb_dolp(path: str | Path, dolp_root: str | Path | None = None, dolp_channels: int = 1) -> np.ndarray:
    """Load aligned RGB and DoLP channels without a depth channel."""
    if dolp_channels not in {1, 3}:
        raise ValueError(f"Expected one or three DoLP channels, got {dolp_channels}")
    path = Path(path)
    rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
    dolp_dir = Path(dolp_root) if dolp_root is not None else path.parent
    dolp_suffix = "dolp_color_rgb" if dolp_channels == 3 else "dolp_rgb"
    dolp_path = dolp_dir / f"{path.stem}_{dolp_suffix}.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    dolp = cv2.imread(str(dolp_path), cv2.IMREAD_COLOR if dolp_channels == 3 else cv2.IMREAD_GRAYSCALE)
    if rgb is None or dolp is None:
        raise FileNotFoundError(f"Missing RGB or DoLP input for {path}: {rgb_path}, {dolp_path}")
    if rgb.shape[:2] != dolp.shape[:2]:
        raise ValueError(f"Paired RGB/DoLP dimensions disagree for {path}")
    image = np.empty((*rgb.shape[:2], 3 + dolp_channels), dtype=np.float32)
    image[..., :3] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    image[..., 3:] = cv2.cvtColor(dolp, cv2.COLOR_BGR2RGB) if dolp_channels == 3 else dolp[..., None]
    return image


def load_rgb_dolp(path: str | Path, dolp_root: str | Path | None = None, dolp_channels: int = 1) -> np.ndarray:
    """Load aligned RGB and one or three DoLP channels."""
    return _load_rgb_dolp(path, dolp_root=dolp_root, dolp_channels=dolp_channels)


def load_rgb_aolp_dolp(
    path: str | Path,
    aolp_root: str | Path | None = None,
    dolp_root: str | Path | None = None,
    aolp_channels: int = 3,
    dolp_channels: int = 3,
) -> np.ndarray:
    """Load aligned RGB, AoLP and DoLP as RGB-first channels."""
    if aolp_channels != 3 or dolp_channels != 3:
        raise ValueError("RGB+AoLP+DoLP currently requires three channels for both polarization maps")
    path = Path(path)
    rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
    aolp_dir = Path(aolp_root) if aolp_root is not None else path.parent
    dolp_dir = Path(dolp_root) if dolp_root is not None else path.parent
    aolp_path = aolp_dir / f"{path.stem}_aolp_rgb.png"
    dolp_path = dolp_dir / f"{path.stem}_dolp_color_rgb.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    aolp = cv2.imread(str(aolp_path), cv2.IMREAD_COLOR)
    dolp = cv2.imread(str(dolp_path), cv2.IMREAD_COLOR)
    if rgb is None or aolp is None or dolp is None:
        raise FileNotFoundError(f"Missing RGB/AoLP/DoLP input for {path}: {rgb_path}, {aolp_path}, {dolp_path}")
    if rgb.shape[:2] != aolp.shape[:2] or rgb.shape[:2] != dolp.shape[:2]:
        raise ValueError(f"Paired RGB/AoLP/DoLP dimensions disagree for {path}")
    image = np.empty((*rgb.shape[:2], 9), dtype=np.float32)
    image[..., :3] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    image[..., 3:6] = cv2.cvtColor(aolp, cv2.COLOR_BGR2RGB)
    image[..., 6:] = cv2.cvtColor(dolp, cv2.COLOR_BGR2RGB)
    return image


def load_rgb_dolp_depth(path: str | Path, dolp_root: str | Path | None = None, dolp_channels: int = 1) -> np.ndarray:
    """Load aligned RGB, DoLP and raw uint16 depth channels."""
    path = Path(path)
    visual = _load_rgb_dolp(path, dolp_root=dolp_root, dolp_channels=dolp_channels)
    depth_path = path.parent.parent / "depth" / f"{path.stem}_depth_dense_u16.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Missing depth input for {path}: {depth_path}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected single-channel uint16 depth at {depth_path}, got {depth.dtype}, {depth.shape}")
    if visual.shape[:2] != depth.shape:
        raise ValueError(f"Paired image dimensions disagree for {path}")
    image = np.empty((*depth.shape, visual.shape[-1] + 1), dtype=np.float32)
    image[..., :-1] = visual
    image[..., -1] = depth
    return image


def normalize_modalities(images: torch.Tensor) -> torch.Tensor:
    """Normalize visual-only or RGB/DoLP/depth BCHW inputs."""
    if images.ndim != 4 or images.shape[1] not in {4, 5, 6, 7, 9}:
        raise ValueError(f"Expected BCHW four- to nine-channel modality input, got {tuple(images.shape)}")
    images = images.float()
    if images.shape[1] in {4, 6, 9}:
        return images / 255.0
    depth = images[:, -1:]
    normalized_depth = torch.where(depth > 0, (depth / 256.0 - DEPTH_MEAN_M) / DEPTH_STD_M, 0.0)
    return torch.cat((images[:, :-1] / 255.0, normalized_depth), dim=1)
