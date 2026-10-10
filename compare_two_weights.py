"""Compare YOLO26 RGB+DoLP and RGB+DoLP+depth weights on every image in a split.

Example:
    python compare_two_weights.py --weights /path/to/pz4/best.pt /path/to/pz5/best.pt \
        --data ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml --output runs/compare_depth
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.pz_modalities import load_rgb_dolp_depth, normalize_modalities
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.detect.val import DetectionValidator
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
    parser.add_argument("--weights", nargs=2, type=Path, required=True, metavar=("NODEPTH", "DEPTH"))
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/compare_two_weights")
    parser.add_argument("--labels-dir", type=Path, help="override YOLO label txt directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16, help="validation batch size")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence for comparison images only")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


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
    images = [p for p in images if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.lower().endswith(AUX_SUFFIXES)]
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
    height = max(30, round(image.shape[0] * 0.045))
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (25, 25, 25), -1)
    cv2.putText(image, title, (8, round(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return image


def infer(model, image, channels, args, device):
    """Use tensor inference so five-channel depth retains its z-score normalization."""
    padded = LetterBox(new_shape=(args.imgsz, args.imgsz), center=True)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0).to(device)
    tensor = normalize_modalities(tensor) if channels == 5 else tensor.float() / 255.0
    with torch.inference_mode():
        raw = model.model(tensor)
        detections = non_max_suppression(raw, args.conf, args.iou, end2end=model.model.model[-1].end2end)[0]
        if len(detections):
            detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], image.shape[:2])
    return [
        {"class_id": int(row[5]), "class_name": str(model.names[int(row[5])]),
         "confidence": row[4], "xyxy": row[:4]}
        for row in detections.cpu().tolist()
    ]


def evaluate(model, weight, name, data, args):
    metrics = model.val(
        validator=PZ5Validator if name == "depth" else DetectionValidator,
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, project=str(args.output / "evaluation"),
        name=name, exist_ok=True, plots=not args.no_plots, verbose=False, fraction=1.0,
    )
    row = {key: float(value) for key, value in metrics.results_dict.items()}
    row.update(weights=str(weight), model=name, split=args.split)
    (args.output / "evaluation" / name / "metrics.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return row


def main(args):
    weights = [path.expanduser().resolve() for path in args.weights]
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
    datasets = []
    for channels, name in ((4, "nodepth"), (5, "depth")):
        config = {**data, "channels": channels, "ch": channels,
                  "rgb_dolp": channels == 4, "rgb_dolp_depth": channels == 5}
        target = args.output / f"eval_data_{name}.yaml"
        YAML.save(target, config)
        datasets.append(target)

    device = select_device(args.device)
    models = [YOLO(str(weight)) for weight in weights]
    for model, channels in zip(models, (4, 5)):
        if model.model.yaml.get("channels") != channels:
            raise ValueError(f"Expected {channels}-channel weights, got {model.model.yaml.get('channels')}")
        model.model.to(device).eval()

    comparisons = args.output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    predictions = []
    for index, path in enumerate(paths, 1):
        input5 = load_rgb_dolp_depth(path)
        rgb = cv2.cvtColor(input5[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        height, width = rgb.shape[:2]
        truth = ground_truth(path, args.labels_dir, names, width, height)
        gt_panel = rgb.copy()
        for class_id, box in truth:
            draw_box(gt_panel, box, class_id, str(names.get(class_id, class_id)))
        panels = [panel(gt_panel, f"label | {len(truth)} objects")]
        record = {"image": str(path), "models": {}}
        for model, channels, name in zip(models, (4, 5), ("nodepth", "depth")):
            detections = infer(model, input5[..., :channels], channels, args, device)
            record["models"][name] = detections
            canvas = rgb.copy()
            for det in detections:
                draw_box(canvas, det["xyxy"], det["class_id"], f'{det["class_name"]} {det["confidence"]:.2f}')
            panels.append(panel(canvas, f"{name} | {len(detections)} detections"))
        output = comparisons / f"{index:06d}_{path.stem}.jpg"
        encoded, content = cv2.imencode(".jpg", np.concatenate(panels, axis=1))
        if not encoded:
            raise OSError(f"Failed to encode comparison: {output}")
        content.tofile(str(output))
        predictions.append(record)
        print(f"[{index}/{len(paths)}] {output}")

    (args.output / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [evaluate(model, weight, name, dataset, args)
            for model, weight, name, dataset in zip(models, weights, ("nodepth", "depth"), datasets)]
    with (args.output / "evaluation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(paths)} comparisons and two validations in {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
