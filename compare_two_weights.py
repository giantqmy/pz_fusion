"""Compare any number of PZ YOLO detection weights alongside ground truth.

Example:
    python compare_two_weights.py --weights /path/to/pz4/best.pt /path/to/hmoe/best.pt \
        --names Baseline HMoE --data ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml \
        --output runs/compare_hmoe --skip-eval
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.pz_modalities import load_rgb_dolp, load_rgb_dolp_depth, normalize_modalities
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.pz import PZHMoEFusion, PZHMoEJointFusion
from ultralytics.utils import YAML, ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device

from train_yolo26_pz5 import PZ5Validator

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
AUX_SUFFIXES = (
    "_s0_rgb", "_s0_nir", "_dolp_rgb", "_dolp_nir", "_aolp_rgb", "_aolp_nir", "_depth", "_depth_dense_u16",
)
COLORS = ((45, 190, 70), (210, 180, 40), (0, 140, 255), (180, 80, 230), (255, 160, 40), (40, 210, 220))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True, metavar="PT")
    parser.add_argument("--names", nargs="+", metavar="NAME", help="optional display names, one per weight")
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/compare_two_weights")
    parser.add_argument("--labels-dir", type=Path, help="override label txt directory for comparison images only")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16, help="validation batch size")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence for comparison images only")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="save comparisons without running full validation")
    args = parser.parse_args()
    if args.names is not None and len(args.names) != len(args.weights):
        parser.error("--names must have exactly one name per weight")
    return args


def source_images(data, split):
    root = Path(data["path"])
    entries = data[split]
    entries = entries if isinstance(entries, list) else [entries]
    images = []
    for entry in entries:
        location = Path(entry)
        if not location.is_absolute():
            location = root / location
        if location.is_dir():
            images.extend(sorted(p for p in location.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        elif location.suffix.lower() in IMAGE_SUFFIXES:
            images.append(location)
        else:
            for line in location.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line)
                images.append(path if path.is_absolute() else (location.parent / path).resolve())
    images = list(dict.fromkeys(
        p.resolve() for p in images if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.lower().endswith(AUX_SUFFIXES)
    ))
    missing = next((p for p in images if not p.is_file()), None)
    if missing:
        raise FileNotFoundError(f"Missing dataset image: {missing}")
    if not images:
        raise ValueError(f"No base images in {split} split")
    return images


def ground_truth(image, labels_dir, names, width, height):
    label = labels_dir / f"{image.stem}.txt" if labels_dir else Path(img2label_paths([str(image)])[0])
    if not label.is_file():
        raise FileNotFoundError(f"Missing ground truth label: {label}")
    boxes = []
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if values:
            class_id, x, y, w, h = map(float, values[:5])
            if class_id != int(class_id) or int(class_id) not in names:
                raise ValueError(f"Invalid class ID {class_id} in {label}")
            boxes.append((int(class_id), ((x - w / 2) * width, (y - h / 2) * height,
                                          (x + w / 2) * width, (y + h / 2) * height)))
    return boxes


def draw_box(canvas, xyxy, class_id, text):
    color = COLORS[class_id % len(COLORS)]
    x1, y1, x2, y2 = (round(float(value)) for value in xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    scale = max(0.45, min(canvas.shape[:2]) / 1000)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    top = max(y1, th + baseline + 3)
    cv2.rectangle(canvas, (x1, top - th - baseline - 3), (x1 + tw + 4, top + 2), color, -1)
    cv2.putText(canvas, text, (x1 + 2, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def panel(image, title):
    """Add a separate two-line header so titles never cover objects in the image."""
    header = np.full((56, image.shape[1], 3), 25, dtype=np.uint8)
    for index, line in enumerate(title.split("\n")[:2]):
        shortened = line
        while shortened and cv2.getTextSize(shortened, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] > image.shape[1] - 16:
            shortened = shortened[:-1]
        cv2.putText(header, shortened, (8, 22 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((header, image), axis=0)


def infer(model, image, channels, args, device):
    """Use tensor inference so five-channel depth retains its z-score normalization."""
    padded = LetterBox(new_shape=(args.imgsz, args.imgsz), center=True)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0).to(device)
    tensor = normalize_modalities(tensor) if channels == 5 else tensor.float() / 255.0
    with torch.inference_mode():
        raw = model.model(tensor)
        detections = non_max_suppression(raw, args.conf, args.iou,
                                        end2end=getattr(model.model.model[-1], "end2end", False))[0]
        if len(detections):
            detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], image.shape[:2])
    return [
        {"class_id": int(row[5]), "class_name": str(model.names[int(row[5])]),
         "confidence": row[4], "xyxy": row[:4]}
        for row in detections.cpu().tolist()
    ]


def evaluate(model, info, data, args):
    metrics = model.val(
        validator=PZ5Validator if info["channels"] == 5 else DetectionValidator,
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, project=str(args.output / "evaluation"),
        name=info["id"], exist_ok=True, plots=not args.no_plots, verbose=False, fraction=1.0,
    )
    row = {key: float(value) for key, value in metrics.results_dict.items()}
    row.update(weights=info["weights"], model=info["id"], name=info["name"],
               channels=info["channels"], modality=info["modality"], split=args.split)
    (args.output / "evaluation" / info["id"] / "metrics.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return row


def main(args):
    args.output = args.output.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    if args.labels_dir:
        args.labels_dir = args.labels_dir.expanduser().resolve()
    weights = [path.expanduser().resolve() for path in args.weights]
    if any(path.suffix.lower() != ".pt" for path in weights):
        raise ValueError("--weights accepts .pt detection checkpoints only")
    for path in (*weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = YAML.load(args.data)
    root = Path(data["path"]).expanduser()
    if not root.is_absolute():
        root = (args.data.resolve().parent / root).resolve()
    data["path"] = str(root)
    paths = source_images(data, args.split)
    names = data["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    # Validation and comparison use the exact same filtered image list.
    manifest = args.output / "eval_images.txt"
    manifest.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    data[args.split] = str(manifest)
    device = select_device(args.device)
    comparisons = args.output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    predictions = [{"image": str(path), "models": {}} for path in paths]
    model_info, rows = [], []
    for index, weight in enumerate(weights, 1):
        model = YOLO(str(weight))
        channels = model.model.yaml.get("channels", model.model.yaml.get("ch"))
        if model.task != "detect" or channels not in {4, 5}:
            raise ValueError(f"Expected a 4/5-channel detection model at {weight}, got {model.task}, channels={channels}")
        model_names = model.names
        model_names = dict(enumerate(model_names)) if isinstance(model_names, list) else {
            int(k): v for k, v in model_names.items()
        }
        if model_names != names:
            raise ValueError(f"Model classes disagree with dataset classes for {weight}: {model_names} != {names}")
        modality = "RGB+DoLP" if channels == 4 else "RGB+DoLP+Depth"
        if any(isinstance(layer, PZHMoEJointFusion) for layer in model.model.modules()):
            modality += " HMoE-Joint"
        elif any(isinstance(layer, PZHMoEFusion) for layer in model.model.modules()):
            modality += " HMoE"
        experiment = weight.parent.parent.name if weight.parent.name == "weights" else weight.parent.name
        info = {"id": f"model_{index:02d}", "name": args.names[index - 1] if args.names else experiment,
                "weights": str(weight), "channels": channels, "modality": modality}
        model_info.append(info)
        print(f"[{index}/{len(weights)} models] {info['name']} | {modality} | {weight}")
        model.model.to(device).float().eval()
        loader = load_rgb_dolp_depth if channels == 5 else load_rgb_dolp
        for image_index, (path, record) in enumerate(zip(paths, predictions), 1):
            record["models"][info["id"]] = infer(model, loader(path), channels, args, device)
            print(f"  [{image_index}/{len(paths)}] {path.name}")
        if not args.skip_eval:
            config = {**data, "channels": channels, "ch": channels,
                      "rgb_dolp": channels == 4, "rgb_dolp_depth": channels == 5}
            target = args.output / f"eval_data_{channels}ch.yaml"
            YAML.save(target, config)
            rows.append(evaluate(model, info, target, args))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for index, (path, record) in enumerate(zip(paths, predictions), 1):
        visual = load_rgb_dolp(path)
        rgb = cv2.cvtColor(visual[..., :3], cv2.COLOR_RGB2BGR)
        height, width = rgb.shape[:2]
        truth = ground_truth(path, args.labels_dir, names, width, height)
        gt_panel = rgb.copy()
        for class_id, box in truth:
            draw_box(gt_panel, box, class_id, str(names.get(class_id, class_id)))
        record["ground_truth"] = [{"class_id": class_id, "class_name": str(names[class_id]), "xyxy": box}
                                  for class_id, box in truth]
        panels = [panel(gt_panel, f"Ground Truth\n{len(truth)} objects")]
        for info in model_info:
            detections = record["models"][info["id"]]
            canvas = rgb.copy()
            for det in detections:
                draw_box(canvas, det["xyxy"], det["class_id"], f'{det["class_name"]} {det["confidence"]:.2f}')
            panels.append(panel(canvas, f"{info['name']}\n{info['modality']} | {len(detections)} detections"))
        output = comparisons / f"{index:06d}_{path.stem}.jpg"
        columns = min(3, len(panels))
        panels.extend(np.zeros_like(panels[0]) for _ in range((-len(panels)) % columns))
        grid = np.concatenate([np.concatenate(panels[i:i + columns], axis=1)
                               for i in range(0, len(panels), columns)], axis=0)
        encoded, content = cv2.imencode(".jpg", grid)
        if not encoded:
            raise OSError(f"Failed to encode comparison: {output}")
        content.tofile(str(output))
        print(f"[{index}/{len(paths)}] {output}")

    (args.output / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "models.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (args.output / "evaluation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(paths)} comparisons for {len(weights)} models and {len(rows)} validations in {args.output}")


if __name__ == "__main__":
    main(parse_args())
