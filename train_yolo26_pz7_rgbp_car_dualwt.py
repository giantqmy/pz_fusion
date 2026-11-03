"""Train the historical RGB/DoLP dual-wavelet model on paired TIFF data."""

from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch

from train_yolo26_pz7_pim_saffconcat_hmoe_joint_no_depth import PZ7NoDepthTrainer
from train_yolo26_pz5 import get_parser, main
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parent
DATASET_CANDIDATES = (Path(r"I:\KEDADATA\rgb-p-car\RGB-P Car"),)
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class RGBPDualWTDataset(YOLODataset):
    """Load RGB and same-stem original DoLP TIFF as a six-channel image."""

    def load_image_file(self, path: str, flags: int | None = None) -> np.ndarray:
        rgb_path = Path(path)
        dolp_path = rgb_path.parent.parent / "dolp" / f"{rgb_path.stem}_dolp.tiff"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        dolp = cv2.imread(str(dolp_path), cv2.IMREAD_COLOR)
        if rgb is None or dolp is None:
            raise FileNotFoundError(f"Missing RGB or original DoLP TIFF: {rgb_path}, {dolp_path}")
        if rgb.shape[:2] != dolp.shape[:2]:
            raise ValueError(f"RGB/DoLP image dimensions disagree: {rgb_path} {rgb.shape}, {dolp_path} {dolp.shape}")
        image = np.concatenate(
            (cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(dolp, cv2.COLOR_BGR2RGB)), axis=2
        )
        if image.shape[2] != 6:
            raise ValueError(f"Expected RGB+3-channel DoLP input, got {image.shape} for {rgb_path}")
        return image


class RGBPDualWTTrainer(PZ7NoDepthTrainer):
    """Keep the historical trainer checks while loading paired TIFF inputs."""

    def build_dataset(self, img_path, mode="train", batch=None):
        self.data["channels"] = 6
        self.data["rgb_dolp_tiff"] = True
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        rect = self.args.rect or mode == "val"
        return RGBPDualWTDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=copy(self.args),
            rect=rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )

    def preprocess_batch(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float() / 255.0
        return batch


def validate_dataset(dataset: Path) -> None:
    """Check train RGB, original DoLP, and YOLO labels without converting files."""
    image_root = dataset / "train" / "images"
    dolp_root = dataset / "train" / "dolp"
    label_root = dataset / "train" / "labels"
    if not image_root.is_dir() or not dolp_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Expected train/images, train/dolp, and train/labels under {dataset}")
    images = [p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise ValueError(f"No training images found under {image_root}")
    for rgb in images:
        dolp = dolp_root / f"{rgb.stem}_dolp.tiff"
        label = label_root / f"{rgb.stem}.txt"
        if not dolp.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing paired original DoLP TIFF or label: {rgb}, {dolp}, {label}")


def resolve_dataset(requested: Path | None) -> Path:
    """Resolve the RGB-P Car root and validate its paired TIFF layout."""
    candidates = (requested,) if requested else DATASET_CANDIDATES
    for path in candidates:
        if (path / "train" / "images").is_dir():
            validate_dataset(path)
            return path
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"RGB-P Car train/images was not found at: {locations}. Pass the root with --dataset.")


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--dataset", type=Path, default=None, help="RGB-P Car dataset root")
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26l-pz7-pim-saffconcat-no-depth.yaml",
        workers=2,
        name="yolo26l_pz7_rgbp_car_dualwt",
        val=False,
    )
    args = parser.parse_args()
    dataset = resolve_dataset(args.dataset)
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")

    import yaml

    model_config = yaml.safe_load(args.model.read_text(encoding="utf-8"))
    model_config["nc"] = 1
    model_config["channels"] = 6
    model_config["ch"] = 6
    for layer in model_config["backbone"]:
        if layer[2] == "PZDualPIMStage":
            layer[2] = "PZDualWTStage"
    dataset_model = ROOT / "runs/rgbp_car/yolo26l-pz7-rgbp-car-dualwt.yaml"
    dataset_model.parent.mkdir(parents=True, exist_ok=True)
    dataset_model.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
    args.model = dataset_model

    dataset_yaml = {
        "path": str(dataset),
        "train": "train/images",
        "val": "test/images",
        "channels": 6,
        "ch": 6,
        "rgb_dolp_tiff": True,
        "dolp_channels": 3,
        "names": {0: "Car"},
    }
    dataset_config = ROOT / "runs/rgbp_car/rgb-p-car-rgb-dolp-prepared.yaml"
    dataset_config.parent.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    args.data = dataset_config
    main(args, trainer_cls=RGBPDualWTTrainer)
