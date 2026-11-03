"""Train YOLO26 with five-channel RGB + DoLP + depth inputs."""

import argparse
from copy import copy
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.data.pz_modalities import normalize_modalities
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

ROOT = Path(__file__).resolve().parent


class PZ5Validator(DetectionValidator):
    def preprocess(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = normalize_modalities(batch["img"], self.data)
        return batch


class PZ5Trainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        self.data.setdefault("channels", 5)
        self.data["rgb_dolp_depth"] = True
        return super().build_dataset(img_path, mode, batch)

    def preprocess_batch(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = normalize_modalities(batch["img"], self.data)
        return batch

    def get_validator(self):
        return PZ5Validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


def main(args, trainer_cls=PZ5Trainer):
    if args.weights and args.resume:
        raise ValueError("Use --weights for a fresh fine-tuning run or --resume for checkpoint recovery, not both.")
    model = YOLO(str(args.resume or args.model))
    if args.weights:
        model.load(str(args.weights))
    train_args = dict(
        trainer=trainer_cls,
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        rect=args.rect,
        project=str(args.project),
        name=args.name,
        pretrained=bool(args.weights) if not args.resume else False,
        freeze=0,
        resume=str(args.resume) if args.resume else False,
        val=getattr(args, "val", True),
    )
    if args.optimizer is not None:
        train_args["optimizer"] = args.optimizer
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0
    if args.amp is not None:
        train_args["amp"] = args.amp
    model.train(**train_args)


def get_parser():
    """Build the five-channel training command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/26/yolo26-pz5.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6qu-rgb-dolp-depth.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rect", action="store_true", help="Use rectangular batches during training.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="yolo26_pz5")
    parser.add_argument("--weights", type=Path, default=None, help="Initialize a fresh run from model weights.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to an interrupted training last.pt checkpoint.")
    parser.add_argument("--optimizer", default=None, help="Optional optimizer override, e.g. AdamW or SGD.")
    parser.add_argument("--lr0", type=float, default=None, help="Optional initial learning rate override.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Override automatic mixed precision.")
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True, help="Run validation during training.")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
