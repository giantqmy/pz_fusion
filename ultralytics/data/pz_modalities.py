"""Paired PZ detection input: RGB, AoLP/DoLP, and raw uint16 depth."""

from pathlib import Path

import cv2
import numpy as np
import torch

DEPTH_MEAN_M = 29.652361724372745
DEPTH_STD_M = 23.830449242515485


def _load_rgb_polarization(
    path: str | Path, polarization_root: str | Path | None, channels: int, polarization: str
) -> np.ndarray:
    """Load aligned RGB and one polarization map in grayscale or color form."""
    if channels not in {1, 3}:
        raise ValueError(f"Expected one or three {polarization.upper()} channels, got {channels}")
    path = Path(path)
    rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
    modality_dir = Path(polarization_root) if polarization_root is not None else path.parent
    suffix = f"{polarization}_color_rgb" if channels == 3 else f"{polarization}_rgb"
    modality_path = modality_dir / f"{path.stem}_{suffix}.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    modality_image = cv2.imread(str(modality_path), cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE)
    if rgb is None or modality_image is None:
        raise FileNotFoundError(f"Missing RGB or {polarization.upper()} input for {path}: {rgb_path}, {modality_path}")
    if rgb.shape[:2] != modality_image.shape[:2]:
        raise ValueError(f"Paired RGB/{polarization.upper()} dimensions disagree for {path}")
    image = np.empty((*rgb.shape[:2], 3 + channels), dtype=np.float32)
    image[..., :3] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    image[..., 3:] = cv2.cvtColor(modality_image, cv2.COLOR_BGR2RGB) if channels == 3 else modality_image[..., None]
    return image


def _load_rgb_dolp(path: str | Path, dolp_root: str | Path | None = None, dolp_channels: int = 1) -> np.ndarray:
    """Load aligned RGB and DoLP channels without a depth channel."""
    return _load_rgb_polarization(path, dolp_root, dolp_channels, "dolp")


def load_rgb_dolp(path: str | Path, dolp_root: str | Path | None = None, dolp_channels: int = 1) -> np.ndarray:
    """Load aligned RGB and one or three DoLP channels."""
    return _load_rgb_dolp(path, dolp_root=dolp_root, dolp_channels=dolp_channels)


def load_rgb_aolp(path: str | Path, aolp_root: str | Path | None = None, aolp_channels: int = 1) -> np.ndarray:
    """Load aligned RGB and AoLP channels without a depth channel."""
    return _load_rgb_polarization(path, aolp_root, aolp_channels, "aolp")


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


def load_rgb_dolp_depth(
    path: str | Path,
    dolp_root: str | Path | None = None,
    dolp_channels: int = 1,
    depth_root: str | Path | None = None,
) -> np.ndarray:
    """Load aligned RGB, DoLP and raw uint16 depth channels."""
    return _load_rgb_polarization_depth(path, dolp_root, dolp_channels, depth_root, "dolp")


def load_rgb_aolp_depth(
    path: str | Path,
    aolp_root: str | Path | None = None,
    aolp_channels: int = 3,
    depth_root: str | Path | None = None,
) -> np.ndarray:
    """Load aligned RGB, color AoLP and raw uint16 depth channels."""
    return _load_rgb_polarization_depth(path, aolp_root, aolp_channels, depth_root, "aolp")


def _load_rgb_polarization_depth(path, polarization_root, polarization_channels, depth_root, polarization):
    """Load a color polarization image with raw uint16 depth as the final channel."""
    path = Path(path)
    visual = _load_rgb_polarization(path, polarization_root, polarization_channels, polarization)
    depth_dir = Path(depth_root) if depth_root is not None else path.parent.parent / "depth"
    depth_path = depth_dir / f"{path.stem}_depth_dense_u16.png"
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


def normalize_modalities(images: torch.Tensor, data: dict | None = None) -> torch.Tensor:
    """Normalize BCHW visual inputs and optional depth using dataset overrides or the training defaults."""
    if images.ndim != 4 or images.shape[1] not in {3, 4, 5, 6, 7, 9}:
        raise ValueError(f"Expected BCHW input with 3, 4, 5, 6, 7 or 9 channels, got {tuple(images.shape)}")
    images = images.float()
    if images.shape[1] in {3, 4, 6, 9}:
        return images / 255.0
    data = data or {}
    scale = float(data.get("depth_scale", 256.0))
    mean = float(data.get("depth_mean", DEPTH_MEAN_M))
    std = float(data.get("depth_std", DEPTH_STD_M))
    depth = images[:, -1:]
    normalized_depth = torch.where(depth > 0, (depth / scale - mean) / std, 0.0)
    return torch.cat((images[:, :-1] / 255.0, normalized_depth), dim=1)
