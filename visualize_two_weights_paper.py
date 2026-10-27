"""Save six-panel Ultralytics-style ground-truth and model comparisons.

The two ``--weights`` are ordered as ``NODEPTH DEPTH``. The RGB-only baseline
is supplied with ``--official-weight``, and YOLO12n with ``--yolo12-weight``.
PCDNet is supplied with ``--pcdnet-weight`` and uses S0 RGB plus scalar AoLP/DoLP.
Each output image contains ground truth, YOLO12n, YOLO26n, PCDNet, no-depth, and
depth results, in that order. The six panels are also saved separately in
``ground_truth/``, ``yolo12n/``, ``rgb_baseline/``, ``rgb_dolp/``, and
``rgb_dolp_depth/``, plus ``pcdnet/`` under the output directory.
All five models are validated on the complete selected split.

Example:
    python visualize_two_weights_paper.py \
        --weights runs/nodepth/weights/best.pt runs/depth/weights/best.pt \
        --data /path/to/your-dataset.yaml --split val \
        --output runs/paper_compare --device 0
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from pcdnet import infer as infer_pcdnet
from pcdnet import load_model as load_pcdnet
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.pz_modalities import load_rgb_dolp, load_rgb_dolp_depth, normalize_modalities
from ultralytics.data.utils import img2label_paths
from ultralytics.engine.results import Results
from ultralytics.utils import YAML, ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device

ROOT = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_WEIGHT = Path(
    "/media/ddc/新加卷/hys/qmy/ultralytics/runs/detect/runs/train/yolo26/weights/best.pt"
)
DEFAULT_YOLO12_WEIGHT = Path(
    "/media/ddc/新加卷/hys/qmy/ultralytics/runs/detect/runs/train/yolo26-2/weights/best.pt"
)
DEFAULT_PCDNET_WEIGHT = Path(
    "/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/pcdnet_s0_aolp_dolp/weights/best.pt"
)
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
AUX_SUFFIXES = ("_s0_rgb", "_s0_nir", "_dolp_rgb", "_dolp_nir", "_aolp_rgb", "_aolp_nir", "_depth")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs=2, type=Path, required=True, metavar=("NODEPTH", "DEPTH"))
    parser.add_argument("--official-weight", type=Path, default=DEFAULT_OFFICIAL_WEIGHT,
                        help="RGB-only YOLO26n baseline checkpoint")
    parser.add_argument("--yolo12-weight", type=Path, default=DEFAULT_YOLO12_WEIGHT,
                        help="RGB-only YOLO12n baseline checkpoint")
    parser.add_argument("--pcdnet-weight", type=Path, default=DEFAULT_PCDNET_WEIGHT,
                        help="PCDNet best.pt from the migrated PZ_8/train_pcdnet.py")
    parser.add_argument("--aolp-root", type=Path, help="PCDNet scalar AoLP directory; default: beside each base image")
    parser.add_argument("--pcdnet-dolp-root", type=Path,
                        help="PCDNet scalar DoLP directory; separate from the color --dolp-root")
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
    parser.add_argument("--val-batch", type=int, default=4, help="Batch size for per-model validation")
    parser.add_argument("--workers", type=int, default=2, help="Data-loader workers for validation")
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
    dolp_dir = resolve_path(dolp_root or data.get("dolp_root", path.parent), data_file.parent)
    if channels == 3 + dolp_channels:
        return load_rgb_dolp(path, dolp_dir, dolp_channels)
    depth_dir = resolve_path(depth_root or data.get("depth_root", path.parent.parent / "depth"), data_file.parent)
    return load_rgb_dolp_depth(path, dolp_dir, dolp_channels, depth_dir)


def normalize_input(image, channels, data):
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()
    return normalize_modalities(tensor, data)


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


def validation_data(data, channels, modality):
    """Build the modality-specific dataset configuration expected by PZ7 val."""
    config = dict(data)
    config["channels"] = channels
    config["ch"] = channels
    config["rgb_s0"] = modality == "rgb_s0"
    config["rgb_dolp"] = modality == "rgb_dolp"
    config["rgb_dolp_depth"] = modality == "rgb_dolp_depth"
    config["dolp_channels"] = dolp_channel_count(data, channels)
    if modality != "rgb_s0":
        expected = 3 + config["dolp_channels"] + int(modality == "rgb_dolp_depth")
        if channels != expected:
            raise ValueError(f"{modality} expects {expected} channels from the dataset config, but model expects {channels}")
    return config


def validate_pcdnet(model, names, data, args, device):
    """Evaluate the complete split at low confidence using the same PCDNet inputs as visualization."""
    from types import SimpleNamespace

    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import DetMetrics, box_iou

    metrics = DetMetrics(names=names)
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    images = source_images(data, args.split, args.data.expanduser().resolve())
    model.to(device).eval()
    for index, path in enumerate(images, 1):
        rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Missing PCDNet S0 input: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        detections = infer_pcdnet(model, path, rgb, args.imgsz, 0.001, 0.65, device,
                                  args.aolp_root, args.pcdnet_dolp_root, multi_label=True)
        truth = label_boxes(path, rgb.shape, args.labels_dir)
        classes = torch.tensor([class_id for class_id, _ in truth], dtype=torch.float32)
        boxes = torch.tensor([box for _, box in truth], dtype=torch.float32).reshape(-1, 4)
        correct = BaseValidator.match_predictions(matcher, detections[:, 5], classes,
                                                  box_iou(boxes, detections[:, :4])).numpy()
        metrics.update_stats({"tp": correct, "conf": detections[:, 4].numpy(),
                              "pred_cls": detections[:, 5].numpy(), "target_cls": classes.numpy(),
                              "target_img": np.unique(classes.numpy()), "im_name": str(path)})
        print(f"val/pcdnet [{index}/{len(images)}]: {path.name}")
    metrics.process()
    model.to("cpu")
    return {key: float(value) for key, value in metrics.results_dict.items()}


def validate_models(models, pcdnet, names, data, args, output, device):
    """Validate the four YOLO models and PCDNet independently on the selected split."""
    from train_yolo26_pz5 import PZ5Validator

    specs = (
        ("yolo12n", models[0], 3, "rgb_s0", None),
        ("official_yolo26n", models[1], 3, "rgb_s0", None),
        ("nodepth", models[2], model_channels(models[2]), "rgb_dolp", PZ5Validator),
        ("depth", models[3], model_channels(models[3]), "rgb_dolp_depth", PZ5Validator),
    )
    metrics_by_model = {}
    validation_root = output / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pz7_val_data_") as temp_dir:
        pcdnet.to("cpu")
        for name, model, channels, modality, validator in specs:
            config = validation_data(data, channels, modality)
            data_yaml = Path(temp_dir) / f"{name}.yaml"
            YAML.save(data_yaml, config)
            for other in models:
                other.model.to("cpu")
            model.model.to(device).eval()
            val_kwargs = {
                "data": str(data_yaml),
                "split": args.split,
                "imgsz": args.imgsz,
                "batch": args.val_batch,
                "workers": args.workers,
                "device": args.device,
                "project": str(validation_root),
                "name": name,
                "exist_ok": True,
                "plots": False,
                "save_json": False,
                "verbose": False,
            }
            if validator is not None:
                val_kwargs["validator"] = validator
            result = model.val(**val_kwargs)
            metrics_by_model[name] = {key: float(value) for key, value in result.results_dict.items()}
            print(f"val/{name}: {json.dumps(metrics_by_model[name], ensure_ascii=False)}")
            model.model.to("cpu")

    metrics_by_model["pcdnet"] = validate_pcdnet(pcdnet, names, data, args, device)
    print(f"val/pcdnet: {json.dumps(metrics_by_model['pcdnet'], ensure_ascii=False)}")
    metrics_file = output / "validation_metrics.json"
    metrics_file.write_text(json.dumps(metrics_by_model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation metrics to {metrics_file}")
    return metrics_by_model


def main(args):
    weights = [p.expanduser().resolve() for p in args.weights]
    official_weight = args.official_weight.expanduser().resolve()
    yolo12_weight = args.yolo12_weight.expanduser().resolve()
    pcdnet_weight = args.pcdnet_weight.expanduser().resolve()
    data_file = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (*weights, official_weight, yolo12_weight, pcdnet_weight, data_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = YAML.load(data_file)
    data["path"] = str(resolve_path(data.get("path", "."), data_file.parent))
    for key, override in (("dolp_root", args.dolp_root), ("depth_root", args.depth_root)):
        value = override or data.get(key)
        if value:
            data[key] = str(resolve_path(value, data_file.parent))
    for attr, key in (("aolp_root", "aolp_root"), ("pcdnet_dolp_root", "pcdnet_dolp_root")):
        value = getattr(args, attr) or data.get(key)
        if value:
            setattr(args, attr, resolve_path(value, data_file.parent))
    images = source_images(data, args.split, data_file)
    if args.max_images > 0:
        images = images[:args.max_images]
    device = select_device(args.device)
    models = [YOLO(str(yolo12_weight)), YOLO(str(official_weight)), *(YOLO(str(weight)) for weight in weights)]
    channels = [model_channels(model) for model in models]
    if channels[:2] != [3, 3]:
        raise ValueError(f"RGB-only YOLO12n and YOLO26n checkpoints must expect 3 input channels, got {channels[:2]}")
    if channels[2] == channels[3]:
        raise ValueError(f"Expected different no-depth/depth input channels, got {channels[2:]}")
    for model in models:
        model.model.to(device).eval()
    names = names_dict(models[2].names)
    for label, model in zip(("YOLO12n", "official YOLO26n", "no-depth", "depth"), models):
        model_names = names_dict(model.names)
        if model_names != names:
            raise ValueError(f"Class names for {label} do not match the no-depth model: {model_names} != {names}")
    pcdnet = load_pcdnet(pcdnet_weight, device)
    if names_dict(pcdnet.names) != names or pcdnet.model[-1].nc != len(names):
        raise ValueError("PCDNet checkpoint class IDs/names must match the comparison dataset")
    print("Panel order: ground truth | YOLO12n | YOLO26n | PCDNet | RGB+DoLP | RGB+DoLP+Depth")

    records = []
    for index, image_path in enumerate(images, 1):
        loaded = [load_modalities(image_path, data, channel, data_file, args.dolp_root, args.depth_root)
                  for channel in channels[2:]]
        rgb_bgr = cv2.cvtColor(loaded[0][..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        truth = label_boxes(image_path, rgb_bgr.shape, args.labels_dir)
        truth_tensor = torch.tensor(
            [[*box, 1.0, class_id] for class_id, box in truth], dtype=torch.float32
        ) if truth else torch.empty((0, 6), dtype=torch.float32)
        gt_canvas = plot_results(rgb_bgr, truth_tensor, names, args.box_width, show_conf=False)
        rgb_input = loaded[0][..., :3]
        yolo12_detections = infer(models[0], rgb_input, data, args, device)
        official_detections = infer(models[1], rgb_input, data, args, device)
        pcdnet_detections = infer_pcdnet(pcdnet, image_path, rgb_input, args.imgsz, args.conf, args.iou,
                                        device, args.aolp_root, args.pcdnet_dolp_root)
        nodepth_detections = infer(models[2], loaded[0], data, args, device)
        depth_detections = infer(models[3], loaded[1], data, args, device)
        yolo12_canvas = plot_results(rgb_bgr, yolo12_detections, names, args.box_width, show_conf=True)
        official_canvas = plot_results(rgb_bgr, official_detections, names, args.box_width, show_conf=True)
        pcdnet_canvas = plot_results(rgb_bgr, pcdnet_detections, names, args.box_width, show_conf=True)
        nodepth_canvas = plot_results(rgb_bgr, nodepth_detections, names, args.box_width, show_conf=True)
        depth_canvas = plot_results(rgb_bgr, depth_detections, names, args.box_width, show_conf=True)

        name = f"{index:06d}_{image_path.stem}.png"
        comparison = np.concatenate(
            (gt_canvas, yolo12_canvas, official_canvas, pcdnet_canvas, nodepth_canvas, depth_canvas), axis=1
        )
        save_image(output / name, comparison)
        save_image(output / "ground_truth" / name, gt_canvas)
        save_image(output / "yolo12n" / name, yolo12_canvas)
        save_image(output / "rgb_baseline" / name, official_canvas)
        save_image(output / "pcdnet" / name, pcdnet_canvas)
        save_image(output / "rgb_dolp" / name, nodepth_canvas)
        save_image(output / "rgb_dolp_depth" / name, depth_canvas)
        records.append({
            "image": str(image_path),
            "ground_truth": [{"class_id": class_id, "xyxy": list(box)} for class_id, box in truth],
            "yolo12n": prediction_records(yolo12_detections, names),
            "official_yolo26n": prediction_records(official_detections, names),
            "pcdnet": prediction_records(pcdnet_detections, names),
            "nodepth": prediction_records(nodepth_detections, names),
            "depth": prediction_records(depth_detections, names),
        })
        gt_summary = ", ".join(names.get(class_id, str(class_id)) for class_id, _ in truth) or "none"
        print(
            f"[{index}/{len(images)}] {name} | "
            f"GT={len(truth)} ({gt_summary}) | "
            f"yolo12n={len(yolo12_detections)} ({terminal_summary(yolo12_detections, names)}) | "
            f"official-yolo26n={len(official_detections)} ({terminal_summary(official_detections, names)}) | "
            f"pcdnet={len(pcdnet_detections)} ({terminal_summary(pcdnet_detections, names)}) | "
            f"nodepth={len(nodepth_detections)} ({terminal_summary(nodepth_detections, names)}) | "
            f"depth={len(depth_detections)} ({terminal_summary(depth_detections, names)})"
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Visualization complete: {len(records)} samples; starting full-split model validation")
    print(f"Saved visualizations and predictions.json to {output}")
    validate_models(models, pcdnet, names, data, args, output, device)


if __name__ == "__main__":
    main(parse_args())
