"""Save named model panels and optional Ultralytics-style ground-truth comparisons.

Repeat ``--net NAME WEIGHTS`` to select models in comparison order. Names become
output subdirectories; use ``pcdnet:NAME`` for a legacy PCDNet checkpoint.
YOLO and RT-DETR checkpoints are detected automatically. Only selected models
are loaded and validated. ``--no-save-comparison`` disables the combined image
while retaining ground-truth/model subdirectories and full-split validation.
Names are printed in the terminal, never added as custom titles to the images.
The legacy ``--weights NODEPTH DEPTH`` command retains its seven-panel defaults.
Each multimodal checkpoint independently selects scalar DoLP (4/5 input channels)
or color DoLP (6/7 input channels). Scalar ``*_dolp_rgb.png`` maps default to the
base image directory when the data YAML describes a different DoLP format.

Example:
    python visualize_two_weights_paper.py \
        --net YOLO11n /path/to/yolo11n/best.pt \
        --net RGB_DoLP /path/to/nodepth/best.pt \
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
from ultralytics import RTDETR, YOLO
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
DEFAULT_RTDETR_WEIGHT = Path(
    "/media/ddc/新加卷/hys/qmy/pz_fusion/runs/train/rtdetr_l_pz_rgb_scratch/weights/best.pt"
)
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
AUX_SUFFIXES = ("_s0_rgb", "_s0_nir", "_dolp_rgb", "_dolp_nir", "_aolp_rgb", "_aolp_nir", "_depth")
SAVE_COMPARISON = True  # CLI: --save-comparison / --no-save-comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--net", nargs=2, action="append", metavar=("NAME", "WEIGHTS"),
                           help="Repeat for each selected model; NAME is its folder/display name. PCDNet: pcdnet:NAME")
    selection.add_argument("--weights", nargs=2, type=Path, metavar=("NODEPTH", "DEPTH"),
                           help="Legacy mode: include both fusion weights and all four baseline defaults")
    parser.add_argument("--official-weight", type=Path, default=DEFAULT_OFFICIAL_WEIGHT,
                        help="RGB-only YOLO26n baseline checkpoint")
    parser.add_argument("--yolo12-weight", type=Path, default=DEFAULT_YOLO12_WEIGHT,
                        help="RGB-only YOLO12n baseline checkpoint")
    parser.add_argument("--pcdnet-weight", type=Path, default=DEFAULT_PCDNET_WEIGHT,
                        help="PCDNet best.pt from the migrated PZ_8/train_pcdnet.py")
    parser.add_argument("--rtdetr-weight", type=Path, default=DEFAULT_RTDETR_WEIGHT,
                        help="RGB-only RT-DETR-L baseline checkpoint")
    parser.add_argument("--aolp-root", type=Path, help="PCDNet scalar AoLP directory; default: beside each base image")
    parser.add_argument("--pcdnet-dolp-root", type=Path,
                        help="PCDNet scalar DoLP directory; separate from the color --dolp-root")
    parser.add_argument("--data", type=Path, required=True, help="Your detection dataset YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--save-comparison", dest="save_comparison", action="store_true",
                            help="Also save the complete horizontal comparison (default)")
    comparison.add_argument("--no-save-comparison", dest="save_comparison", action="store_false",
                            help="Save individual panels only; final validation still runs")
    parser.set_defaults(save_comparison=SAVE_COMPARISON)
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


def dolp_channel_count(channels):
    """Derive scalar/color DoLP from the checkpoint's actual input channels."""
    if channels in {4, 5}:
        return 1
    if channels in {6, 7}:
        return 3
    raise ValueError(f"Expected a 4/5-channel scalar or 6/7-channel color DoLP checkpoint, got {channels}")


def load_modalities(path, data, channels, data_file, dolp_root=None, depth_root=None):
    """Load RGB, DoLP and optional depth in the channel order expected by a weight."""
    modality = "rgb_dolp" if channels in {4, 6} else "rgb_dolp_depth"
    config = validation_data(data, channels, modality, dolp_root)
    dolp_channels = config["dolp_channels"]
    dolp_dir = resolve_path(config.get("dolp_root", path.parent), data_file.parent)
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


def load_models(args, device):
    """Resolve ordered named models, preserving the legacy command's folders and JSON keys."""
    if args.net:
        specs = []
        for name, weight in args.net:
            kind = "pcdnet" if name.lower().startswith("pcdnet:") else "native"
            name = name.split(":", 1)[1] if kind == "pcdnet" else name
            specs.append((name, name, name, Path(weight), kind))
    else:
        specs = [
            ("YOLO12n", "yolo12n", "yolo12n", args.yolo12_weight, "native"),
            ("YOLO26n", "rgb_baseline", "official_yolo26n", args.official_weight, "native"),
            ("RT-DETR-L", "rtdetr_l", "rtdetr_l", args.rtdetr_weight, "native"),
            ("PCDNet", "pcdnet", "pcdnet", args.pcdnet_weight, "pcdnet"),
            ("RGB+DoLP", "rgb_dolp", "nodepth", args.weights[0], "native"),
            ("RGB+DoLP+Depth", "rgb_dolp_depth", "depth", args.weights[1], "native"),
        ]
    used = {"ground_truth", "image", "validation", "predictions.json", "validation_metrics.json"}
    for _, folder, _, weight, _ in specs:
        if (not folder.strip() or folder in {".", ".."} or folder.endswith((".", " "))
                or any(char in folder for char in '<>:"/\\|?*') or folder.casefold() in used):
            raise ValueError(f"Model names must be unique, valid directory names (ground_truth/validation reserved): {folder!r}")
        used.add(folder.casefold())
        if not weight.expanduser().is_file():
            raise FileNotFoundError(weight)
    entries = []
    for name, folder, key, weight, kind in specs:
        weight = weight.expanduser().resolve()
        model = load_pcdnet(weight, device) if kind == "pcdnet" else YOLO(str(weight))
        channels = 9 if kind == "pcdnet" else model_channels(model)
        if kind != "pcdnet":
            kind = "rtdetr" if isinstance(model, RTDETR) else "yolo"
            if channels not in {3, 4, 5, 6, 7} or (kind == "rtdetr" and channels != 3):
                raise ValueError(f"Unsupported {name} input: {kind}, {channels} channels")
            model.model.to(device).eval()
        entries.append(dict(name=name, folder=folder, key=key, model=model, kind=kind, channels=channels))
        print(f"Model {name}: {kind}, {channels} channels; {weight}")
    return entries


def load_rgb(path):
    """Read the dataset's S0 RGB independently of any optional polarization/depth inputs."""
    rgb_path = path.with_name(f"{path.stem}_S0_rgb.png")
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing S0 RGB input: {rgb_path}")
    return image


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


def print_class_ap(model_name, box_metrics, names):
    """Print AP50 and AP50-95 for every class, marking classes absent from val as N/A."""
    class_ap = {
        int(class_id): values
        for class_id, values in zip(box_metrics.ap_class_index, box_metrics.all_ap)
    }
    rows = []
    for class_id, class_name in names.items():
        values = class_ap.get(class_id)
        ap50 = f"{values[0]:.4f}" if values is not None else "N/A"
        ap5095 = f"{values.mean():.4f}" if values is not None else "N/A"
        rows.append(f"{class_name}: AP50={ap50}, AP50-95={ap5095}")
    print(f"val/{model_name} per-class AP: " + " | ".join(rows))


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def validation_data(data, channels, modality, dolp_root=None):
    """Select the checkpoint's DoLP format and use identical roots for visualization and val."""
    config = dict(data)
    config["channels"] = channels
    config["ch"] = channels
    config["rgb_s0"] = modality == "rgb_s0"
    config["rgb_dolp"] = modality == "rgb_dolp"
    config["rgb_dolp_depth"] = modality == "rgb_dolp_depth"
    config["rgb_aolp_dolp"] = False
    if modality != "rgb_s0":
        config["dolp_channels"] = dolp_channel_count(channels)
        if int(data.get("dolp_channels", config["dolp_channels"])) != config["dolp_channels"]:
            config.pop("dolp_root", None)
        if dolp_root is not None:
            config["dolp_root"] = str(dolp_root)
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
        rgb = cv2.cvtColor(load_rgb(path), cv2.COLOR_BGR2RGB)
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
    print_class_ap("pcdnet", metrics.box, names)
    model.to("cpu")
    return {key: float(value) for key, value in metrics.results_dict.items()}


def validate_models(entries, names, data, args, output, device):
    """Validate every selected model on the complete split with its matching input modalities."""
    from train_yolo26_pz5 import PZ5Validator

    metrics_by_model = {}
    validation_root = output / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pz7_val_data_") as temp_dir:
        for entry in entries:
            name, model, channels = entry["key"], entry["model"], entry["channels"]
            for other in entries:
                (other["model"] if other["kind"] == "pcdnet" else other["model"].model).to("cpu")
            if entry["kind"] == "pcdnet":
                metrics_by_model[name] = validate_pcdnet(model, names, data, args, device)
                print(f"val/{entry['name']}: {json.dumps(metrics_by_model[name], ensure_ascii=False)}")
                continue
            modality = "rgb_s0" if channels == 3 else "rgb_dolp" if channels in {4, 6} else "rgb_dolp_depth"
            config = validation_data(data, channels, modality, data.get("dolp_root") if args.dolp_root else None)
            data_yaml = Path(temp_dir) / f"{entry['folder']}.yaml"
            YAML.save(data_yaml, config)
            model.model.to(device).eval()
            val_kwargs = {
                "data": str(data_yaml),
                "split": args.split,
                "imgsz": args.imgsz,
                "batch": args.val_batch,
                "workers": args.workers,
                "device": args.device,
                "project": str(validation_root),
                "name": entry["folder"],
                "exist_ok": True,
                "plots": False,
                "save_json": False,
                "verbose": False,
            }
            if channels != 3:
                val_kwargs["validator"] = PZ5Validator
            result = model.val(**val_kwargs)
            metrics_by_model[name] = {key: float(value) for key, value in result.results_dict.items()}
            print(f"val/{entry['name']}: {json.dumps(metrics_by_model[name], ensure_ascii=False)}")
            print_class_ap(entry["name"], result.box, names)
            model.model.to("cpu")

    metrics_file = output / "validation_metrics.json"
    metrics_file.write_text(json.dumps(metrics_by_model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation metrics to {metrics_file}")
    return metrics_by_model


def main(args):
    data_file = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
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
    entries = load_models(args, device)
    names = names_dict(entries[0]["model"].names)
    for entry in entries:
        model_names = names_dict(entry["model"].names)
        if model_names != names:
            raise ValueError(f"Class IDs/names for {entry['name']} do not match: {model_names} != {names}")
    print("Panel order: ground truth | " + " | ".join(entry["name"] for entry in entries))
    print(f"Save complete comparison: {args.save_comparison}")

    records = []
    for index, image_path in enumerate(images, 1):
        rgb_bgr = load_rgb(image_path)
        rgb_input = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        truth = label_boxes(image_path, rgb_bgr.shape, args.labels_dir)
        truth_tensor = torch.tensor(
            [[*box, 1.0, class_id] for class_id, box in truth], dtype=torch.float32
        ) if truth else torch.empty((0, 6), dtype=torch.float32)
        gt_canvas = plot_results(rgb_bgr, truth_tensor, names, args.box_width, show_conf=False)
        name = f"{index:06d}_{image_path.stem}.png"
        save_image(output / "ground_truth" / name, gt_canvas)
        record = {
            "image": str(image_path),
            "ground_truth": [{"class_id": class_id, "xyxy": list(box)} for class_id, box in truth],
        }
        canvases = [gt_canvas]
        loaded = {3: rgb_input}
        for entry in entries:
            model, channels = entry["model"], entry["channels"]
            if entry["kind"] == "pcdnet":
                detections = infer_pcdnet(model, image_path, rgb_input, args.imgsz, args.conf, args.iou,
                                          device, args.aolp_root, args.pcdnet_dolp_root)
            elif entry["kind"] == "rtdetr":
                # RT-DETR owns its square stretch and decoder postprocessing (no YOLO NMS).
                result = model.predict(source=rgb_bgr, imgsz=args.imgsz, conf=args.conf,
                                       device=args.device, max_det=300, save=False, verbose=False)[0]
                detections = result.boxes.data.detach().cpu()
            else:
                if channels not in loaded:
                    loaded[channels] = load_modalities(image_path, data, channels, data_file,
                                                       args.dolp_root, args.depth_root)
                detections = infer(model, loaded[channels], data, args, device)
            canvas = plot_results(rgb_bgr, detections, names, args.box_width, show_conf=True)
            save_image(output / entry["folder"] / name, canvas)
            if args.save_comparison:
                canvases.append(canvas)
            record[entry["key"]] = prediction_records(detections, names)
        if args.save_comparison:
            save_image(output / name, np.concatenate(canvases, axis=1))
        records.append(record)
        print(f"[{index}/{len(images)}] Saved {name}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Visualization complete: {len(records)} samples; starting full-split model validation")
    print(f"Saved visualizations and predictions.json to {output}")
    validate_models(entries, names, data, args, output, device)


if __name__ == "__main__":
    main(parse_args())
