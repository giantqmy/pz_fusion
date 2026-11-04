"""Compare YOLO and PCDNet checkpoints on RGB-P Car using original TIFF inputs.

Expected layout::

    DATASET_ROOT/{train,test}/{images,aolp,dolp,labels}/
    images/<stem>.tiff
    aolp/<stem>_aolp.tiff
    dolp/<stem>_dolp.tiff
    labels/<stem>.txt

Repeat ``--net NAME WEIGHTS`` to choose models and panel order. Use
``--net pcdnet:NAME WEIGHTS`` for a migrated PCDNet checkpoint. Three-channel
YOLO models receive RGB; six-channel YOLO models receive RGB concatenated with
the original three-channel DoLP TIFF. PCDNet receives original RGB, AoLP and
DoLP TIFFs. No pseudo-color polarization PNG is read.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pcdnet import infer as infer_pcdnet
from pcdnet import load_model as load_pcdnet
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.pz_modalities import normalize_modalities
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--net", nargs=2, action="append", required=True, metavar=("NAME", "WEIGHTS"),
        help="Model name and checkpoint; repeat to add models in comparison order.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="RGB-P Car dataset root")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aolp-root", type=Path, help="Override the original AoLP TIFF directory")
    parser.add_argument("--dolp-root", type=Path, help="Override the original DoLP TIFF directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--box-width", type=int, default=3)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--save-comparison", dest="save_comparison", action="store_true")
    comparison.add_argument("--no-save-comparison", dest="save_comparison", action="store_false")
    parser.set_defaults(save_comparison=True)
    return parser.parse_args()


def resolve_original_tiff(root: Path, stem: str, modality: str):
    candidates = (root / f"{stem}_{modality}.tiff", root / f"{stem}_{modality}.tif")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing original {modality.upper()} TIFF: {candidates[0]}")
    return path


def get_images(
    dataset: Path,
    split: str,
    max_images: int,
    require_dolp: bool,
    require_aolp: bool,
    dolp_dir: Path,
    aolp_dir: Path,
):
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"
    required_dirs = [image_dir, label_dir]
    if require_dolp:
        required_dirs.append(dolp_dir)
    if require_aolp:
        required_dirs.append(aolp_dir)
    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected RGB-P Car directory: {directory}")
    paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.lower().endswith("_dolp")
    )
    if not paths:
        raise ValueError(f"No RGB images found in {image_dir}")
    for path in paths:
        label = label_dir / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing YOLO label for {path}: {label}")
        if require_dolp:
            resolve_original_tiff(dolp_dir, path.stem, "dolp")
        if require_aolp:
            resolve_original_tiff(aolp_dir, path.stem, "aolp")
    return paths[:max_images] if max_images > 0 else paths


def load_pair(rgb_path: Path, dolp_dir: Path):
    dolp_path = resolve_original_tiff(dolp_dir, rgb_path.stem, "dolp")
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    dolp_bgr = cv2.imread(str(dolp_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None or dolp_bgr is None:
        raise OSError(f"Could not decode original TIFF pair: {rgb_path}, {dolp_path}")
    if rgb_bgr.shape[:2] != dolp_bgr.shape[:2]:
        raise ValueError(f"RGB/DoLP dimensions differ: {rgb_path} {rgb_bgr.shape}, {dolp_path} {dolp_bgr.shape}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    dolp = cv2.cvtColor(dolp_bgr, cv2.COLOR_BGR2RGB)
    return rgb_bgr, rgb, np.concatenate((rgb, dolp), axis=2)


def model_channels(model):
    channels = model.model.yaml.get("channels", model.model.yaml.get("ch"))
    if channels is None:
        channels = model.model.model[0].conv.in_channels
    channels = int(channels)
    if channels not in {3, 6}:
        raise ValueError(f"This script supports RGB (3-channel) or RGB+original color DoLP (6-channel) models, got {channels}")
    return channels


def load_models(specs, device):
    entries = []
    used_names = {"ground_truth", "predictions.json"}
    for raw_name, weight_value in specs:
        kind = "pcdnet" if raw_name.lower().startswith("pcdnet:") else "yolo"
        name = raw_name.split(":", 1)[1] if kind == "pcdnet" else raw_name
        if not name.strip() or name.casefold() in used_names or Path(name).name != name:
            raise ValueError(f"Invalid model name for output directory: {name!r}")
        used_names.add(name.casefold())
        weight = Path(weight_value).expanduser().resolve()
        if not weight.is_file():
            raise FileNotFoundError(weight)
        model = load_pcdnet(weight, device) if kind == "pcdnet" else YOLO(str(weight))
        channels = 9 if kind == "pcdnet" else model_channels(model)
        if kind == "yolo":
            model.model.to(device).eval()
        names = model.names
        names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else {int(k): str(v) for k, v in names.items()}
        entries.append({
            "name": name,
            "model": model,
            "kind": kind,
            "channels": channels,
            "names": names,
            "weight": weight,
        })
        input_description = "RGB + original AoLP/DoLP TIFF" if kind == "pcdnet" else f"{channels}-channel"
        print(f"Model {name}: {kind}, {input_description} input; {weight}")
    baseline_names = entries[0]["names"]
    if any(
        entry["names"].keys() != baseline_names.keys()
        or any(entry["names"][class_id].casefold() != baseline_names[class_id].casefold() for class_id in baseline_names)
        for entry in entries[1:]
    ):
        raise ValueError("All checkpoints must use the same class IDs and names for a comparison panel.")
    return entries


def read_labels(path: Path, shape):
    height, width = shape[:2]
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected YOLO label rows with 5 values in {path}: {line!r}")
        class_id, x, y, box_w, box_h = map(float, fields)
        boxes.append((int(class_id), ((x - box_w / 2) * width, (y - box_h / 2) * height,
                                      (x + box_w / 2) * width, (y + box_h / 2) * height)))
    return boxes


def infer(model, image, channels, imgsz, conf, iou, device):
    padded = LetterBox(new_shape=(imgsz, imgsz), center=True)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
    tensor = normalize_modalities(tensor, {}).to(device)
    with torch.inference_mode():
        raw = model.model(tensor[:, :channels])
        head = model.model.model[-1]
        detections = non_max_suppression(
            raw, conf, iou, end2end=getattr(head, "end2end", False)
        )[0]
    if len(detections):
        detections = detections.clone()
        detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], image.shape[:2])
    return detections.detach().cpu()


def draw(image, boxes, names, box_width, show_conf):
    result = Results(image, path="", names=names, boxes=boxes)
    return result.plot(conf=show_conf, line_width=box_width, labels=True, boxes=True, color_mode="class")


def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def main(args):
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    entries = load_models(args.net, device)
    names = entries[0]["names"]
    dolp_dir = (args.dolp_root or dataset / args.split / "dolp").expanduser().resolve()
    aolp_dir = (args.aolp_root or dataset / args.split / "aolp").expanduser().resolve()
    require_dolp = any(entry["channels"] == 6 or entry["kind"] == "pcdnet" for entry in entries)
    require_aolp = any(entry["kind"] == "pcdnet" for entry in entries)
    images = get_images(
        dataset,
        args.split,
        args.max_images,
        require_dolp,
        require_aolp,
        dolp_dir,
        aolp_dir,
    )
    print("Panel order: ground truth | " + " | ".join(entry["name"] for entry in entries))
    if require_dolp:
        print("DoLP source: original TIFF files in " + str(dolp_dir))
    if require_aolp:
        print("AoLP source: original TIFF files in " + str(aolp_dir))

    records = []
    for index, rgb_path in enumerate(images, 1):
        if require_dolp:
            rgb_bgr, rgb, rgb_dolp = load_pair(rgb_path, dolp_dir)
        else:
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise OSError(f"Could not decode RGB TIFF: {rgb_path}")
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            rgb_dolp = None
        label_path = dataset / args.split / "labels" / f"{rgb_path.stem}.txt"
        truth = read_labels(label_path, rgb_bgr.shape)
        truth_boxes = torch.tensor(
            [[*box, 1.0, class_id] for class_id, box in truth], dtype=torch.float32
        ) if truth else torch.empty((0, 6), dtype=torch.float32)
        canvas_gt = draw(rgb_bgr, truth_boxes, names, args.box_width, show_conf=False)
        file_name = f"{index:06d}_{rgb_path.stem}.png"
        save_image(output / "ground_truth" / file_name, canvas_gt)
        canvases = [canvas_gt]
        record = {
            "image": str(rgb_path),
            "rgb_tiff": str(rgb_path),
            "ground_truth": [{"class_id": cid, "xyxy": list(box)} for cid, box in truth],
        }
        if require_dolp:
            record["dolp_tiff"] = str(resolve_original_tiff(dolp_dir, rgb_path.stem, "dolp"))
        if require_aolp:
            record["aolp_tiff"] = str(resolve_original_tiff(aolp_dir, rgb_path.stem, "aolp"))
        for entry in entries:
            if entry["kind"] == "pcdnet":
                detections = infer_pcdnet(
                    entry["model"],
                    rgb_path,
                    rgb,
                    args.imgsz,
                    args.conf,
                    args.iou,
                    device,
                    aolp_dir,
                    dolp_dir,
                    tiff_only=True,
                )
            else:
                model_input = rgb if entry["channels"] == 3 else rgb_dolp
                detections = infer(
                    entry["model"], model_input, entry["channels"],
                    args.imgsz, args.conf, args.iou, device
                )
            canvas = draw(rgb_bgr, detections, names, args.box_width, show_conf=True)
            save_image(output / entry["name"] / file_name, canvas)
            if args.save_comparison:
                canvases.append(canvas)
            record[entry["name"]] = [
                {"class_id": int(row[5]), "class_name": names[int(row[5])], "confidence": float(row[4]),
                 "xyxy": [float(value) for value in row[:4]]}
                for row in detections.tolist()
            ]
        if args.save_comparison:
            save_image(output / file_name, np.concatenate(canvases, axis=1))
        records.append(record)
        print(f"[{index}/{len(images)}] Saved {file_name}")

    predictions = output / "predictions.json"
    predictions.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Visualization complete: {len(records)} images; results saved to {output}")
    print(f"Predictions saved to {predictions}")


if __name__ == "__main__":
    main(parse_args())
