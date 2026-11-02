"""Train randomly initialized YOLO11l on RGB-P Car S0 RGB images only."""

import argparse
from copy import copy
from pathlib import Path

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_split_fraction, img2label_paths
from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

ROOT = Path(__file__).resolve().parent


class RGBPCarS0Dataset(YOLODataset):
    """Map S0 RGB image names to labels named after the base capture timestamp."""

    def get_label_files(self) -> list[str]:
        label_files = img2label_paths(self.im_files)
        for index, image_path in enumerate(self.im_files):
            stem = Path(image_path).stem
            if stem.endswith("_S0_rgb"):
                label_path = Path(label_files[index])
                label_files[index] = str(label_path.with_name(f"{stem[:-len('_S0_rgb')]}{label_path.suffix}"))
        self.label_files = label_files
        return label_files


def build_rgbpcar_dataset(cfg, img_path, batch, data, mode, stride, fraction):
    """Build the normal Ultralytics detection dataset with the RGB-P Car label mapping."""
    fraction = get_split_fraction(fraction, mode)
    return RGBPCarS0Dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=copy(cfg),
        rect=cfg.rect or mode == "val",
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
    )


class RGBPCarS0Validator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        fraction = get_split_fraction(self.args.fraction, self.args.split or "val")
        return build_rgbpcar_dataset(self.args, img_path, batch, self.data, mode, self.stride, fraction)


class RGBPCarS0Trainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_rgbpcar_dataset(self.args, img_path, batch, self.data, mode, stride, self.args.fraction)

    def get_validator(self):
        return RGBPCarS0Validator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/rgb-p-car-rgb-s0.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="yolo11l_rgbpcar_s0_scratch")
    args = parser.parse_args()

    data = YAML.load(args.data)
    if data.get("channels", 3) != 3 or any(
        data.get(key) for key in ("rgb_dolp", "rgb_dolp_depth", "rgb_aolp_dolp")
    ):
        parser.error("--data must describe RGB-only, three-channel input")

    model = YOLO(str(ROOT / "ultralytics/cfg/models/11/yolo11l.yaml"))
    model.train(
        trainer=RGBPCarS0Trainer,
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=False,
        resume=False,
        optimizer="SGD",
        lr0=args.lr0,
        lrf=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        cos_lr=True,
        warmup_bias_lr=0.0,
        amp=False,
        deterministic=False,
        seed=args.seed,
        patience=0,
        val=True,
        save=True,
        plots=True,
        cache=False,
    )


if __name__ == "__main__":
    main()
