"""Save Ultralytics-style ground-truth and two-model detection visualizations.

The two weights are ordered as ``NODEPTH DEPTH``. Bounding boxes are rendered
through :class:`ultralytics.engine.results.Results`, so class colors, labels,
and confidence formatting match the official Ultralytics visualization.

Example:
    python visualize_two_weights_paper.py \
        --weights runs/nodepth/weights/best.pt runs/depth/weights/best.pt \
        --data /path/to/your-dataset.yaml --split val \
        --output runs/paper_compare --device 0
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import img2label_paths
from ultralytics.engine.results import Results
from ultralytics.utils import YAML, ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
AUX_SUFFIXES = ("_s0_rgb", "_s0_nir", "_dolp_rgb", "_dolp_nir", "_aolp_rgb", "_aolp_nir", "_depth")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs=2, type=Path, required=True, metavar=("NODEPTH", "DEPTH"))
    parser.add_argument("--data", type=Path, required=True, help="Your detection dataset YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, help="Optional flat directory containing YOLO label txt files")
    parser.add_argument("--dolp-root", type=Path, help="Override data YAML dolp_root")
    parser.add_argument("--depth-root", type=Path, help="Override the inferred depth directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--box-width", type=int, default=3)
    return parser.parse_args()


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def source_images(data, split, data_file):
    root = resolve_path(data.get("path", "."), data_file.parent)
    entries = data.get(split)
    if entries is None:
        raise KeyError(f"Dataset YAML has no '{split}' entry: {data_file}")
    entries = entries if isinstance(entries, list) else [entries]
    images = []
    for entry in entries:
        location = resolve_path(entry, root)
        if location.is_dir():
            images.extend(sorted(p for p in location.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        elif location.suffix.lower() in IMAGE_SUFFIXES:
            images.append(location)
        elif location.is_file():
            for line in location.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line)
                images.append(path if path.is_absolute() else (location.parent / path).resolve())
        else:
            raise FileNotFoundError(f"Dataset split entry does not exist: {location}")
    images = [p for p in images if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.lower().endswith(AUX_SUFFIXES)]
    missing = next((p for p in images if not p.is_file()), None)
    if missing:
        raise FileNotFoundError(f"Missing dataset image: {missing}")
    if not images:
        raise ValueError(f"No images found in '{split}' split")
    return images


def _first_existing(paths, description):
    for path in paths:
        if path.is_file():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing {description}; tried: {joined}")


def dolp_channel_count(data, channels):
    """Return the DoLP channel count, defaulting to color DoLP for 6/7-channel models."""
    return int(data.get("dolp_channels", 3 if channels in {6, 7, 9} else 1))


def load_modalities(path, data, channels, data_file, dolp_root=None, depth_root=None):
    """Load RGB, DoLP and optional depth in the channel order expected by a weight."""
    dolp_channels = dolp_channel_count(data, channels)
    if channels not in {3 + dolp_channels, 4 + dolp_channels}:
        raise ValueError(
            f"Weight expects {channels} channels, but data/dolp_channels imply {3 + dolp_channels} visual "
            f"or {4 + dolp_channels} RGB-DoLP-depth channels"
        )
    rgb_path = _first_existing(
        (path.with_name(f"{path.stem}_S0_rgb.png"), path), "RGB image"
    )
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise OSError(f"Could not decode RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    dolp_dir = resolve_path(dolp_root or data.get("dolp_root", path.parent), data_file.parent)
    color_candidates = (
        dolp_dir / f"{path.stem}_dolp_color_rgb.png",
        dolp_dir / f"{path.stem}_dolp_rgb.png",
    )
    dolp_path = _first_existing(color_candidates, "DoLP image")
    dolp = cv2.imread(str(dolp_path), cv2.IMREAD_COLOR if dolp_channels == 3 else cv2.IMREAD_GRAYSCALE)
    if dolp is None:
        raise OSError(f"Could not decode DoLP image: {dolp_path}")
    if dolp_channels == 3:
        dolp = cv2.cvtColor(dolp, cv2.COLOR_BGR2RGB)
    else:
        dolp = dolp[..., None]

    if rgb.shape[:2] != dolp.shape[:2]:
        raise ValueError(f"RGB and DoLP dimensions disagree for {path}")
    visual = np.concatenate((rgb, dolp), axis=2).astype(np.float32, copy=False)
    if channels == 3 + dolp_channels:
        return visual

    inferred_depth_root = path.parent.parent / "depth"
    depth_dir = resolve_path(depth_root or data.get("depth_root", inferred_depth_root), data_file.parent)
    depth_candidates = (
        depth_dir / f"{path.stem}_depth_dense_u16.png",
        depth_dir / f"{path.stem}_depth.png",
        depth_dir / f"{path.stem}.png",
    )
    depth_path = _first_existing(depth_candidates, "depth image")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"Expected one-channel depth image: {depth_path}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB and depth dimensions disagree for {path}")
    return np.concatenate((visual, depth[..., None].astype(np.float32)), axis=2)


def normalize_input(image, channels, data):
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()
    visual_channels = channels
    has_depth = channels == 4 + dolp_channel_count(data, channels)
    if has_depth:
        visual_channels -= 1
    tensor[:, :visual_channels] /= 255.0
    if has_depth:
        scale = float(data.get("depth_scale", 256.0))
        mean = float(data.get("depth_mean", 29.652361724372745))
        std = float(data.get("depth_std", 23.830449242515485))
        depth = tensor[:, -1:]
        tensor[:, -1:] = torch.where(depth > 0, (depth / scale - mean) / std, torch.zeros_like(depth))
    return tensor


def model_channels(model):
    channels = model.model.yaml.get("channels", model.model.yaml.get("ch"))
    if channels is None:
        raise ValueError("Could not determine input channels from model YAML")
    return int(channels)


def infer(model, image, data, args, device):
    channels = model_channels(model)
    padded = LetterBox(new_shape=(args.imgsz, args.imgsz), center=True)(image=image)
    tensor = normalize_input(padded, channels, data).to(device)
    with torch.inference_mode():
        raw = model.model(tensor)
        head = model.model.model[-1]
        detections = non_max_suppression(raw, args.conf, args.iou, end2end=getattr(head, "end2end", False))[0]
    if len(detections):
        # Ultralytics returns inference tensors here; scale_boxes updates its input in-place.
        detections = detections.clone()
        detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], image.shape[:2])
    return detections.detach().cpu()


def label_boxes(image_path, image_shape, labels_dir):
    label = labels_dir / f"{image_path.stem}.txt" if labels_dir else Path(img2label_paths([str(image_path)])[0])
    if not label.is_file():
        raise FileNotFoundError(f"Missing ground-truth label: {label}")
    height, width = image_shape[:2]
    boxes = []
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id, x, y, box_w, box_h = map(float, values[:5])
        boxes.append((int(class_id), ((x - box_w / 2) * width, (y - box_h / 2) * height,
                                      (x + box_w / 2) * width, (y + box_h / 2) * height)))
    return boxes


def names_dict(names):
    """Normalize model class names to the mapping expected by ``Results``."""
    return dict(enumerate(names)) if isinstance(names, (list, tuple)) else {int(k): str(v) for k, v in names.items()}


def plot_results(image, boxes, names, line_width, show_conf):
    """Use the official Ultralytics renderer for class colors and box labels."""
    result = Results(image, path="", names=names, boxes=boxes)
    return result.plot(
        conf=show_conf,
        line_width=line_width,
        labels=True,
        boxes=True,
        masks=False,
        probs=False,
        color_mode="class",
    )


def prediction_records(detections, names):
    """Convert native detection rows to JSON-serializable records."""
    return [
        {
            "class_id": int(row[5]),
            "class_name": names[int(row[5])],
            "confidence": float(row[4]),
            "xyxy": [float(v) for v in row[:4]],
        }
        for row in detections.tolist()
    ]


def terminal_summary(detections, names):
    """Format detections for the terminal without adding text to saved images."""
    if not len(detections):
        return "none"
    return ", ".join(f"{names[int(row[5])]} {float(row[4]):.2f}" for row in detections.tolist())


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def main(args):
    weights = [p.expanduser().resolve() for p in args.weights]
    data_file = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (*weights, data_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = YAML.load(data_file)
    images = source_images(data, args.split, data_file)
    if args.max_images > 0:
        images = images[:args.max_images]
    device = select_device(args.device)
    models = [YOLO(str(weight)) for weight in weights]
    channels = [model_channels(model) for model in models]
    if channels[0] == channels[1]:
        raise ValueError(f"Expected different no-depth/depth input channels, got {channels}")
    for model in models:
        model.model.to(device).eval()
    names = names_dict(models[0].names)

    records = []
    for index, image_path in enumerate(images, 1):
        loaded = [load_modalities(image_path, data, channel, data_file, args.dolp_root, args.depth_root)
                  for channel in channels]
        rgb_bgr = cv2.cvtColor(loaded[0][..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        truth = label_boxes(image_path, rgb_bgr.shape, args.labels_dir)
        truth_tensor = torch.tensor(
            [[*box, 1.0, class_id] for class_id, box in truth], dtype=torch.float32
        ) if truth else torch.empty((0, 6), dtype=torch.float32)
        gt_canvas = plot_results(rgb_bgr, truth_tensor, names, args.box_width, show_conf=False)
        nodepth_detections = infer(models[0], loaded[0], data, args, device)
        depth_detections = infer(models[1], loaded[1], data, args, device)
        nodepth_canvas = plot_results(rgb_bgr, nodepth_detections, names, args.box_width, show_conf=True)
        depth_canvas = plot_results(rgb_bgr, depth_detections, names, args.box_width, show_conf=True)

        name = f"{index:06d}_{image_path.stem}.png"
        save_image(output / "ground_truth" / name, gt_canvas)
        save_image(output / "nodepth" / name, nodepth_canvas)
        save_image(output / "depth" / name, depth_canvas)
        save_image(output / "triptych" / name, np.concatenate((gt_canvas, nodepth_canvas, depth_canvas), axis=1))
        records.append({
            "image": str(image_path),
            "ground_truth": [{"class_id": class_id, "xyxy": list(box)} for class_id, box in truth],
            "nodepth": prediction_records(nodepth_detections, names),
            "depth": prediction_records(depth_detections, names),
        })
        gt_summary = ", ".join(names.get(class_id, str(class_id)) for class_id, _ in truth) or "none"
        print(
            f"[{index}/{len(images)}] {name} | "
            f"GT={len(truth)} ({gt_summary}) | "
            f"nodepth={len(nodepth_detections)} ({terminal_summary(nodepth_detections, names)}) | "
            f"depth={len(depth_detections)} ({terminal_summary(depth_detections, names)})"
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validation complete: {len(records)} samples")
    print(f"Saved visualizations and predictions.json to {output}")


if __name__ == "__main__":
    main(parse_args())
