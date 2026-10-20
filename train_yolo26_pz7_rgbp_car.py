"""Train the RGB/color-DoLP PZ7 model on RGB-P Car."""

import json
from pathlib import Path

import cv2
import numpy as np

from train_yolo26_pz7_pim_saffconcat_hmoe_joint_no_depth import PZ7NoDepthTrainer
from train_yolo26_pz5 import get_parser, main

ROOT = Path(__file__).resolve().parent
DATASET_CANDIDATES = (
    Path("/media/ddc/新加卷/hys/qmy/pz/RGB-P Car"),
    Path(r"I:\KEDADATA\rgb-p-car\RGB-P Car"),
)


def resolve_dataset():
    """Select the server dataset path, falling back to the local copy."""
    for path in DATASET_CANDIDATES:
        if (path / "train" / "annotations.json").is_file() and (path / "test" / "annotations.json").is_file():
            return path
    locations = ", ".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"RGB-P Car train/test annotations not found at: {locations}")


def prepare_dataset(dataset):
    """Convert RGB-P COCO annotations and color DoLP into paired YOLO inputs."""
    split_lists = {"train": [], "test": []}

    for split in ("train", "test"):
        source = dataset / split
        coco = json.loads((source / "annotations.json").read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"].lower() for item in coco["categories"]}
        car_ids = {cid for cid, name in categories.items() if name == "car"}
        if len(car_ids) != 1:
            raise ValueError(f"Expected exactly one car category in {split}/annotations.json: {categories}")
        car_id = next(iter(car_ids))
        annotations = {}
        for ann in coco["annotations"]:
            if ann["category_id"] == car_id:
                annotations.setdefault(ann["image_id"], []).append(ann["bbox"])

        candidates = []
        for record in coco["images"]:
            image_id = record["id"]
            image_path = source / "images" / record["file_name"]
            stem = image_path.stem
            dolp_path = source / "dolp" / f"{stem}_dolp.tiff"
            label_path = source / "labels" / f"{stem}.txt"
            if not image_path.is_file() or not dolp_path.is_file():
                raise FileNotFoundError(f"Missing paired RGB/DoLP input: {image_path}, {dolp_path}")
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label file: {label_path}")

            rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            dolp = cv2.imread(str(dolp_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or dolp is None or rgb.shape[:2] != dolp.shape[:2]:
                raise ValueError(f"Invalid or misaligned RGB/DoLP pair: {image_path}, {dolp_path}")
            if rgb.shape[1] != record["width"] or rgb.shape[0] != record["height"]:
                raise ValueError(f"COCO dimensions disagree with image: {image_path}")
            dolp_gray = cv2.normalize(dolp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            dolp_color = cv2.applyColorMap(dolp_gray, cv2.COLORMAP_JET)
            fused_root = dataset / "images" / "rgb_dolp" / split
            label_root = dataset / "labels" / "rgb_dolp" / split
            fused_root.mkdir(parents=True, exist_ok=True)
            label_root.mkdir(parents=True, exist_ok=True)
            fused_stem = f"{split}_{stem}"
            fused_image = fused_root / f"{fused_stem}.tiff"
            if not cv2.imwrite(str(fused_root / f"{fused_stem}_S0_rgb.png"), rgb):
                raise OSError(f"Failed to write RGB image: {fused_stem}")
            if not cv2.imwrite(str(fused_root / f"{fused_stem}_dolp_color_rgb.png"), dolp_color):
                raise OSError(f"Failed to write color DoLP image: {fused_stem}")
            label_lines = []
            width, height = record["width"], record["height"]
            for x, y, w, h in annotations.get(image_id, []):
                label_lines.append(f"0 {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}")
            (label_root / f"{fused_stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
            candidates.append(str(fused_image.resolve()))

        split_lists[split] = candidates

    for name, paths in split_lists.items():
        (dataset / f"{name}.txt").write_text("\n".join(paths), encoding="utf-8")


if __name__ == "__main__":
    dataset = resolve_dataset()
    prepare_dataset(dataset)
    parser = get_parser()
    model_path = ROOT / "ultralytics/cfg/models/26/yolo26-pz7-pim-saffconcat-no-depth.yaml"
    parser.set_defaults(
        model=model_path,
        data=ROOT / "ultralytics/cfg/datasets/rgb-p-car-rgb-dolp.yaml",
        workers=2,
        name="yolo26_pz7_rgbp_car_rgb_dolp",
        val=False,
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    import yaml

    model_config = yaml.safe_load(args.model.read_text(encoding="utf-8"))
    model_config["nc"] = 1
    dataset_model = ROOT / "runs/rgbp_car/yolo26-pz7-rgbp-car.yaml"
    dataset_model.parent.mkdir(parents=True, exist_ok=True)
    dataset_model.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
    args.model = dataset_model
    dataset_yaml = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_yaml["path"] = str(dataset)
    dataset_yaml["dolp_root"] = str(dataset / "images" / "rgb_dolp")
    dataset_config = ROOT / "runs/rgbp_car/rgb-p-car-rgb-dolp.yaml"
    dataset_config.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    args.data = dataset_config
    main(args, trainer_cls=PZ7NoDepthTrainer)
