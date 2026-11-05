"""Load the migrated PCDNet checkpoint without a separate PZ_8 checkout.

Model sources are from compare/AAAI24-PCDNet; see LICENSE. The custom unpickler
maps the original models/utils module names into this isolated package.
"""

import math
import pickle
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from .utils.general import non_max_suppression


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        if module.startswith(("models.", "utils.")):
            module = "pcdnet." + module
        return super().find_class(module, name)


def load_checkpoint(weights, device="cpu"):
    """Load a legacy full-object PCDNet checkpoint with isolated module names."""
    legacy_pickle = types.ModuleType("pcdnet_pickle")
    legacy_pickle.Unpickler = _Unpickler
    legacy_pickle.load = pickle.load
    try:
        return torch.load(weights, map_location=device, pickle_module=legacy_pickle, weights_only=False)
    except TypeError:  # PyTorch versions without the weights_only argument.
        return torch.load(weights, map_location=device, pickle_module=legacy_pickle)


def load_model(weights, device):
    """Restore the full trained model, retaining every checkpoint parameter."""
    checkpoint = load_checkpoint(weights)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a PCDNet checkpoint dictionary containing model or ema")
    model = next((checkpoint.get(key) for key in ("ema", "model")
                  if isinstance(checkpoint.get(key), torch.nn.Module)), None)
    if model is None:
        raise ValueError("PCDNet requires best.pt/last.pt containing the full model from train_pcdnet.py")
    model = model.module if hasattr(model, "module") else model
    model = model.float().to(device).eval()
    head = model.model[-1]
    head.grid = [torch.empty(0, device=device) for _ in range(head.nl)]
    return model


def prepare_inputs(image_path, rgb, imgsz, aolp_root=None, dolp_root=None, tiff_only=False):
    """Load three RGB domains expected by PCDNet, optionally requiring original TIFF inputs."""
    polarization_images = []
    stem = image_path.stem
    if stem.lower().endswith("_s0_rgb"):
        stem = stem[:-len("_S0_rgb")]
    for root, suffix in ((aolp_root, "aolp"), (dolp_root, "dolp")):
        modality_root = root if root is not None else image_path.parent
        tiff_candidates = (
            modality_root / f"{stem}_{suffix}.tiff",
            modality_root / f"{stem}_{suffix}.tif",
        )
        candidates = tiff_candidates if tiff_only else (
            modality_root / f"{stem}_{suffix}_rgb.png",
            *tiff_candidates,
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Missing PCDNet {suffix} input: {path}")
        if image.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"PCDNet modality dimensions disagree: {path}")
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] >= 3:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Expected grayscale or color PCDNet {suffix} input, got {image.shape}: {path}")
        polarization_images.append(image)
    image = np.concatenate((rgb, *polarization_images), axis=-1).astype(np.float32)
    height, width = image.shape[:2]
    ratio = imgsz / max(height, width)
    resized_w, resized_h = min(math.ceil(width * ratio), imgsz), min(math.ceil(height * ratio), imgsz)
    if ratio != 1:
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_h, resized_w = height, width
    pad_x, pad_y = round((imgsz - resized_w) / 2 - 0.1), round((imgsz - resized_h) / 2 - 0.1)
    # Every PCDNet input is visual; the generic five-channel letterbox treats its last channel as depth.
    padded = np.full((imgsz, imgsz, image.shape[2]), 114, dtype=np.float32)
    padded[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = image
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0) / 255.0
    inputs = [tensor[:, :3], tensor[:, 3:6], tensor[:, 6:9]]
    return inputs, (resized_w / width, resized_h / height, pad_x, pad_y)


@torch.inference_mode()
def infer(
    model,
    image_path,
    rgb,
    imgsz,
    conf,
    iou,
    device,
    aolp_root=None,
    dolp_root=None,
    multi_label=False,
    tiff_only=False,
):
    """Decode PCDNet objectness/class scores and restore boxes to the original RGB image."""
    if imgsz % max(int(model.stride.max()), 32):
        raise ValueError("PCDNet imgsz must be a multiple of its maximum stride")
    inputs, (gain_x, gain_y, pad_x, pad_y) = prepare_inputs(
        image_path, rgb, imgsz, aolp_root, dolp_root, tiff_only=tiff_only
    )
    predictions, _ = model([image.to(device) for image in inputs])
    detections = non_max_suppression(predictions, conf_thres=conf, iou_thres=iou, multi_label=multi_label)[0]
    detections = detections.detach().float().cpu().clone()
    if len(detections):
        height, width = rgb.shape[:2]
        detections[:, [0, 2]] = ((detections[:, [0, 2]] - pad_x) / gain_x).clamp(0, width)
        detections[:, [1, 3]] = ((detections[:, [1, 3]] - pad_y) / gain_y).clamp(0, height)
    return detections
