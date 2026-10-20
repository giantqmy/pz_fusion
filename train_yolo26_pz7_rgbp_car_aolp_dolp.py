"""Train the RGB/AoLP/DoLP PZ7 model on RGB-P Car."""

import json
from pathlib import Path

import cv2
import numpy as np

from train_yolo26_pz7_pim_saffconcat_hmoe_joint_no_depth import PZ7NoDepthTrainer
from train_yolo26_pz5 import ROOT, get_parser, main
from ultralytics.models.yolo.detect import DetectionTrainer


DATASET_CANDIDATES = (
    Path("/media/ddc/新加卷/hys/qmy/pz/RGB-P Car"),
    Path(r"I:\KEDADATA\rgb-p-car\RGB-P Car"),
)


class PZ7AoLPDoLPTrainer(PZ7NoDepthTrainer):
    """Train on nine-channel RGB, AoLP and DoLP inputs fused at model layer 0."""

    def build_dataset(self, img_path, mode="train", batch=None):
        self.data["channels"] = 9
        self.data["rgb_aolp_dolp"] = True
        self.data.pop("rgb_dolp", None)
        self.data.pop("rgb_dolp_depth", None)
        return DetectionTrainer.build_dataset(self, img_path, mode, batch)


def resolve_dataset():
    for path in DATASET_CANDIDATES:
        if (path / "train" / "annotations.json").is_file() and (path / "test" / "annotations.json").is_file():
            return path
    locations = ", ".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"RGB-P Car train/test annotations not found at: {locations}")


def prepare_dataset(dataset: Path):
    """Materialize RGB/AoLP/DoLP companions and YOLO labels for the paired loader."""
    split_lists = {"train": [], "test": []}
    for split in split_lists:
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

        image_root = dataset / "images" / "rgb_aolp_dolp" / split
        aolp_root = dataset / "images" / "rgb_aolp_dolp" / "aolp"
        dolp_root = dataset / "images" / "rgb_aolp_dolp" / "dolp"
        label_root = dataset / "labels" / "rgb_aolp_dolp" / split
        for directory in (image_root, aolp_root, dolp_root, label_root):
            directory.mkdir(parents=True, exist_ok=True)

        candidates = []
        for record in coco["images"]:
            image_id = record["id"]
            image_path = source / "images" / record["file_name"]
            stem = image_path.stem
            aolp_path = source / "aolp" / f"{stem}_aolp.tiff"
            dolp_path = source / "dolp" / f"{stem}_dolp.tiff"
            label_path = source / "labels" / f"{stem}.txt"
            if not image_path.is_file() or not aolp_path.is_file() or not dolp_path.is_file():
                raise FileNotFoundError(f"Missing paired RGB/AoLP/DoLP input: {image_path}, {aolp_path}, {dolp_path}")
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label file: {label_path}")

            rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            aolp = cv2.imread(str(aolp_path), cv2.IMREAD_COLOR)
            dolp = cv2.imread(str(dolp_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or aolp is None or dolp is None or rgb.shape[:2] != aolp.shape[:2] or rgb.shape[:2] != dolp.shape[:2]:
                raise ValueError(f"Invalid or misaligned RGB/AoLP/DoLP pair: {image_path}, {aolp_path}, {dolp_path}")
            if rgb.shape[1] != record["width"] or rgb.shape[0] != record["height"]:
                raise ValueError(f"COCO dimensions disagree with image: {image_path}")

            dolp_gray = cv2.normalize(dolp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            dolp_color = cv2.applyColorMap(dolp_gray, cv2.COLORMAP_JET)
            fused_stem = f"{split}_{stem}"
            fused_image = image_root / f"{fused_stem}.tiff"
            outputs = (
                (fused_image, rgb),
                (image_root / f"{fused_stem}_S0_rgb.png", rgb),
                (aolp_root / f"{fused_stem}_aolp_rgb.png", aolp),
                (dolp_root / f"{fused_stem}_dolp_color_rgb.png", dolp_color),
            )
            for output_path, image in outputs:
                if not cv2.imwrite(str(output_path), image):
                    raise OSError(f"Failed to write paired input: {output_path}")

            width, height = record["width"], record["height"]
            lines = [
                f"0 {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}"
                for x, y, w, h in annotations.get(image_id, [])
            ]
            (label_root / f"{fused_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            candidates.append(str(fused_image.resolve()))
        split_lists[split] = candidates

    for split, paths in split_lists.items():
        (dataset / f"{split}_aolp_dolp.txt").write_text("\n".join(paths), encoding="utf-8")


if __name__ == "__main__":
    dataset = resolve_dataset()
    prepare_dataset(dataset)
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz7-aolp-dolp-fusepol-no-depth.yaml",
        data=ROOT / "ultralytics/cfg/datasets/rgb-p-car-aolp-dolp.yaml",
        workers=2,
        name="yolo26_pz7_rgbp_car_aolp_dolp_fusepol",
        val=False,
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    import yaml

    model_config = yaml.safe_load(args.model.read_text(encoding="utf-8"))
    model_config["nc"] = 1
    model_config_path = ROOT / "runs/rgbp_car/yolo26-pz7-aolp-dolp-fusepol.yaml"
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    model_config_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
    args.model = model_config_path

    dataset_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_config["path"] = str(dataset)
    paired_root = dataset / "images" / "rgb_aolp_dolp"
    dataset_config["aolp_root"] = str(paired_root / "aolp")
    dataset_config["dolp_root"] = str(paired_root / "dolp")
    dataset_config_path = ROOT / "runs/rgbp_car/rgb-p-car-aolp-dolp.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")
    args.data = dataset_config_path
    main(args, trainer_cls=PZ7AoLPDoLPTrainer)
