"""Train randomly initialized YOLO11n with the same PZ RGB settings as the RT-DETR-L baseline."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parent


def main():
    """Run RGB-only YOLO11n training and validation with the RT-DETR-L baseline settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz-rgb-only.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="yolo11n_pz_rgb_scratch")
    args = parser.parse_args()
    data = YAML.load(args.data)
    if data.get("channels", 3) != 3 or any(
        data.get(key) for key in ("rgb_dolp", "rgb_dolp_depth", "rgb_aolp_dolp")
    ):
        parser.error("--data must describe RGB-only, three-channel input")

    # The n suffix selects the nano scale of yolo11.yaml, with randomly initialized weights.
    model = YOLO(str(ROOT / "ultralytics/cfg/models/11/yolo11n.yaml"))
    model.train(
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
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        momentum=0.9,
        weight_decay=1e-4,
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
