"""Train the YOLO26 RGB+DoLP ablation without depth input or depth gating."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main(args):
    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        pretrained=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/26/yolo26-pz4.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="yolo26_pz4")
    main(parser.parse_args())
