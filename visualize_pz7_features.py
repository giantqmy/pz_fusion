# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize PZ7 PIM, SAFFConcat and HMoE features for one paired test image.

The input path is the *base* image path used by ``pz_modalities``.  For example,
for ``scene_001.jpg`` the script looks for ``scene_001_S0_rgb.png``, the color
DoLP image and ``depth/scene_001_depth_dense_u16.png`` using the same mapping as
the training dataset.

Example::

    python visualize_pz7_features.py \
        "D:/data/pz/images/val/scene_001.jpg" \
        --weights "D:/runs/yolo26_pz7/weights/best.pt" \
        --dolp-root "D:/data/pz/DOLP_rgb" \
        --output "D:/runs/feature_visualization" --imgsz 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Keep this script runnable from any working directory while using the checked-out
# Ultralytics package that contains the PZ7 custom modules.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_base_path(path: Path) -> Path:
    """Convert a directly selected ``*_S0_rgb`` file into the dataset base path."""
    if path.stem.endswith("_S0_rgb"):
        return path.with_name(path.stem[: -len("_S0_rgb")] + path.suffix)
    return path


def _device(value: str):
    """Resolve ``auto``, a CUDA index, or a normal torch device string."""
    import torch

    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        value = "cpu"
    return torch.device(value)


def _resize_and_normalize(image: np.ndarray, imgsz: int):
    """Resize the raw HWC modalities and apply the exact PZ normalization."""
    import torch
    import torch.nn.functional as F

    if image.ndim != 3 or image.shape[-1] not in {5, 7}:
        raise ValueError(f"Expected HWC five/seven-channel input, got {image.shape}")
    raw = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
    if raw.shape[-2:] != (imgsz, imgsz):
        visual = F.interpolate(raw[:, :-1], size=(imgsz, imgsz), mode="bilinear", align_corners=False)
        depth = F.interpolate(raw[:, -1:], size=(imgsz, imgsz), mode="nearest")
        raw = torch.cat((visual, depth), dim=1)
    from ultralytics.data.pz_modalities import normalize_modalities

    return normalize_modalities(raw)


def _to_feature(value: Any):
    """Detach a feature output, retaining tuples/lists of branch tensors."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_to_feature(item) for item in value)
    return value


def _save_feature_map(feature: Any, directory: Path, block_size: int, cmap_name: str, max_channels: int):
    """Save one tensor (or a tuple/list of tensors) as nearest-neighbor color blocks."""
    from PIL import Image
    from matplotlib import cm

    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(feature, (tuple, list)):
        for index, branch in enumerate(feature):
            _save_feature_map(branch, directory / f"branch_{index}", block_size, cmap_name, max_channels)
        return
    if not hasattr(feature, "ndim"):
        return
    array = np.asarray(feature)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected CHW/BCHW feature map, got {array.shape}")
    channels, height, width = array.shape
    count = channels if max_channels <= 0 else min(channels, max_channels)
    cmap = cm.get_cmap(cmap_name)
    for channel in range(count):
        fmap = np.nan_to_num(array[channel], nan=0.0, posinf=0.0, neginf=0.0)
        low, high = float(fmap.min()), float(fmap.max())
        normalized = (fmap - low) / (high - low + 1e-5)
        rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb).resize((width * block_size, height * block_size), Image.Resampling.NEAREST)
        image.save(directory / f"channel_{channel:04d}.png")
    (directory / "README.txt").write_text(
        f"shape={tuple(array.shape)}\nsaved_channels={count}\n",
        encoding="utf-8",
    )


def _feature_array(feature: Any) -> np.ndarray:
    """Convert a captured BCHW/CHW tensor to a finite CHW NumPy array."""
    if hasattr(feature, "detach"):
        feature = feature.detach().float().cpu().numpy()
    array = np.asarray(feature)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected CHW/BCHW feature map, got {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _colorize_channel(channel: np.ndarray, cmap, low=None, high=None) -> np.ndarray:
    low = float(channel.min()) if low is None else low
    high = float(channel.max()) if high is None else high
    normalized = np.clip((channel - low) / (high - low + 1e-6), 0, 1)
    return (cmap(normalized)[..., :3] * 255).astype(np.uint8)


def _save_channel_contact_sheet(
    feature: Any,
    path: Path,
    block_size: int,
    cmap_name: str,
    comparison: Any | None = None,
    title: str = "",
):
    """Tile every channel into one readable sheet; comparisons share per-channel scaling."""
    from PIL import Image, ImageDraw
    from matplotlib import cm

    arrays = [_feature_array(feature)]
    if comparison is not None:
        arrays.append(_feature_array(comparison))
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError(f"Cannot compare feature maps with different shapes: {[a.shape for a in arrays]}")

    channels, height, width = arrays[0].shape
    columns = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels / columns))
    cell_w, cell_h = width * block_size, height * block_size
    gap, header_h, row_label_w = 3, 22, 58 if comparison is not None else 0
    sheet_w = row_label_w + columns * cell_w + (columns + 1) * gap
    sheet_h = header_h + len(arrays) * (rows * cell_h + (rows + 1) * gap)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((gap, 4), f"{title} | C={channels}, HxW={height}x{width}", fill=(20, 20, 20))
    cmap = cm.get_cmap(cmap_name)

    for row_index, array in enumerate(arrays):
        row_top = header_h + row_index * (rows * cell_h + (rows + 1) * gap)
        if comparison is not None:
            row_name = "Before" if row_index == 0 else "After"
            draw.text((gap, row_top + 4), row_name, fill=(20, 20, 20))
        for channel_index in range(channels):
            grid_y, grid_x = divmod(channel_index, columns)
            x = row_label_w + gap + grid_x * cell_w
            y = row_top + gap + grid_y * cell_h
            if comparison is not None:
                joint = np.concatenate([item[channel_index].ravel() for item in arrays])
                low, high = float(joint.min()), float(joint.max())
                color = _colorize_channel(array[channel_index], cmap, low, high)
            else:
                color = _colorize_channel(array[channel_index], cmap)
            tile = Image.fromarray(color).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            sheet.paste(tile, (x, y))
            draw.text((x + 2, y + 2), str(channel_index), fill=(255, 255, 255), stroke_width=1,
                      stroke_fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_paper_views(features: dict[str, Any], directory: Path, block_size: int, cmap_name: str):
    """Save P2 PIM before/after modality comparisons and SAFF output."""
    levels = ("p2",)
    for level in levels:
        before_rgb = features.get(f"pim/{level}/before_rgb")
        before_dolp = features.get(f"pim/{level}/before_dolp")
        after = features.get(f"pim/{level}/after_branches")
        if before_rgb is not None and before_dolp is not None and isinstance(after, (tuple, list)) and len(after) == 2:
            _save_channel_contact_sheet(
                before_rgb, directory / f"{level}_rgb_pim_before_after.png", block_size, cmap_name,
                comparison=after[0], title=f"{level.upper()} RGB PIM",
            )
            _save_channel_contact_sheet(
                before_dolp, directory / f"{level}_dolp_pim_before_after.png", block_size, cmap_name,
                comparison=after[1], title=f"{level.upper()} DoLP PIM",
            )

        saff = features.get(f"saff/{level}/after_single_fuse")
        if saff is not None:
            _save_channel_contact_sheet(
                saff, directory / f"{level}_saff_fused.png", block_size, cmap_name,
                title=f"{level.upper()} SAFFConcat fused",
            )


def _save_inputs(image: np.ndarray, directory: Path, cmap_name: str):
    """Save RGB, color DoLP, raw depth and a colorized depth preview."""
    from PIL import Image
    from matplotlib import cm

    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image[..., :3], 0, 255).astype(np.uint8)).save(directory / "rgb.png")
    Image.fromarray(np.clip(image[..., 3:-1], 0, 255).astype(np.uint8)).save(directory / "dolp_rgb.png")
    depth = np.asarray(image[..., -1], dtype=np.uint16)
    Image.fromarray(depth).save(directory / "depth_raw_u16.png")
    valid = depth > 0
    scaled = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(depth[valid], (1, 99))
        scaled = np.clip((depth.astype(np.float32) - low) / (high - low + 1e-6), 0, 1)
    Image.fromarray((cm.get_cmap(cmap_name)(scaled)[..., :3] * 255).astype(np.uint8)).save(directory / "depth_color.png")


class FeatureRecorder:
    """Register hooks on PZ7 modules and retain only detached CPU feature maps."""

    def __init__(self, model):
        self.features: dict[str, Any] = {}
        self.handles = []
        self._attach(model)

    def _put(self, name: str, value: Any):
        self.features[name] = _to_feature(value)

    def _attach(self, model):
        layers = getattr(model, "model", model)
        layers = getattr(layers, "model", layers)
        stage_numbers = {"PZDualPIMStage": 0, "PZSAFFConcatLateral": 0, "PZHMoEJointFusion": 0}
        pyramid_levels = ("p2", "p3", "p4", "p5")
        for index, module in enumerate(layers):
            name = module.__class__.__name__
            if name == "PZDualPIMStage":
                level = pyramid_levels[stage_numbers[name]]
                stage_numbers[name] += 1
                self.handles.extend(
                    [
                        module.branch1.register_forward_hook(
                            lambda _m, _i, out, n=f"pim/{level}/before_rgb": self._put(n, out)
                        ),
                        module.branch2.register_forward_hook(
                            lambda _m, _i, out, n=f"pim/{level}/before_dolp": self._put(n, out)
                        ),
                        module.pim.register_forward_hook(
                            lambda _m, _i, out, n=f"pim/{level}/after_branches": self._put(n, out)
                        ),
                        module.register_forward_hook(
                            lambda _m, _i, out, n=f"pim/{level}/after_concat": self._put(n, out)
                        ),
                    ]
                )
            elif name == "PZSAFFConcatLateral":
                level = pyramid_levels[stage_numbers[name]]
                stage_numbers[name] += 1
                self.handles.extend(
                    [
                        module.saff.register_forward_hook(
                            lambda _m, _i, out, n=f"saff/{level}/after_saff_branches": self._put(n, out)
                        ),
                        module.register_forward_hook(
                            lambda _m, _i, out, n=f"saff/{level}/after_single_fuse": self._put(n, out)
                        ),
                    ]
                )
            elif name == "PZHMoEJointFusion":
                level = pyramid_levels[stage_numbers[name]]
                stage_numbers[name] += 1

                def before(_module, inputs, n=f"hmoe/{level}/before_fusion"):
                    self._put(n, inputs[0] if len(inputs) == 1 else inputs)

                self.handles.extend(
                    [
                        module.register_forward_pre_hook(before),
                        module.register_forward_hook(
                            lambda _m, _i, out, n=f"hmoe/{level}/after_joint_fusion": self._put(n, out)
                        ),
                    ]
                )

    def close(self):
        for handle in self.handles:
            handle.remove()


def _level_name(name: str) -> str:
    """Make hook keys readable and stable in the output tree."""
    return name.replace("/", "__")


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Absolute path to the base test image (or *_S0_rgb.png).")
    parser.add_argument("--weights", type=Path, required=True, help="Trained PZ7 .pt checkpoint (best.pt/last.pt).")
    parser.add_argument("--dolp-root", type=Path, default=None, help="Directory containing color DoLP PNG files.")
    parser.add_argument("--dolp-channels", type=int, choices=(3,), default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/feature_visualization")
    parser.add_argument("--imgsz", type=int, default=512, help="Square inference size; use the training size.")
    parser.add_argument("--block-size", type=int, default=8, help="Output pixels per feature-map pixel.")
    parser.add_argument("--max-channels", type=int, default=0, help="Maximum channels per map; 0 saves all channels.")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main(args):
    """Load one image, run the network once and save all requested visualizations."""
    import torch
    from ultralytics import YOLO
    from ultralytics.data.pz_modalities import load_rgb_dolp_depth

    image_path = _as_base_path(args.image.expanduser().resolve())
    if not image_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {image_path}")
    if args.weights.suffix.lower() != ".pt":
        raise ValueError("--weights must point to the trained .pt checkpoint, not the training .py script")
    weights = args.weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    if args.imgsz <= 0 or args.block_size <= 0:
        raise ValueError("--imgsz and --block-size must be positive")

    raw = load_rgb_dolp_depth(image_path, dolp_root=args.dolp_root, dolp_channels=args.dolp_channels)
    output = args.output.expanduser().resolve() / image_path.stem
    _save_inputs(raw, output / "inputs", args.cmap)
    batch = _resize_and_normalize(raw, args.imgsz)

    device = _device(args.device)
    yolo = YOLO(str(weights))
    network = yolo.model.to(device).eval()
    recorder = FeatureRecorder(network)
    try:
        with torch.inference_mode():
            network(batch.to(device))
    finally:
        recorder.close()

    if not recorder.features:
        raise RuntimeError(
            "No PZ7 hooks fired. Check that the checkpoint uses the PZ7 PIM/SAFFConcat/HMoE YAML and local modules."
        )
    for name, feature in recorder.features.items():
        _save_feature_map(
            feature,
            output / "features" / _level_name(name),
            args.block_size,
            args.cmap,
            args.max_channels,
        )
    paper_dir = output / "paper"
    _save_paper_views(recorder.features, paper_dir, args.block_size, args.cmap)
    print(f"Saved {len(recorder.features)} feature groups to: {output}")
    print(f"Saved PIM/SAFF all-channel comparison sheets to: {paper_dir}")


if __name__ == "__main__":
    main(parse_args())
