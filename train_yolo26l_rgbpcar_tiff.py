"""Train a YOLO26-L RGB baseline on the original RGB-P Car TIFF images."""

import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/media/ddc/新加卷/hys/qmy/pz/RGB-P Car")
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def validate_split(dataset: Path, split: str) -> None:
    """Require original RGB images and same-stem YOLO labels for one split."""
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}")
    images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No RGB images found in {image_dir}")
    missing = [label_dir / f"{path.stem}.txt" for path in images if not (label_dir / f"{path.stem}.txt").is_file()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} same-stem labels; examples: {examples}")
    print(f"{split}: {len(images)} original RGB images, labels verified")


def write_configs(dataset: Path) -> tuple[Path, Path]:
    """Create explicit one-class YOLO26-L and RGB TIFF dataset configurations."""
    output_dir = ROOT / "runs" / "rgbp_car"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_model = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
    model_config = yaml.safe_load(source_model.read_text(encoding="utf-8"))
    model_config.update({"nc": 1, "channels": 3, "ch": 3, "scale": "l"})
    model_path = output_dir / "yolo26l-rgbpcar-rgb-tiff.yaml"
    model_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

    data_config = {
        "path": str(dataset),
        "train": "train/images",
        "val": "test/images",
        "channels": 3,
        "names": {0: "Car"},
    }
    data_path = output_dir / "rgb-p-car-rgb-tiff.yaml"
    data_path.write_text(yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return model_path, data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.1, help="Final LR factor; 0.1 makes 0.01 decay to 0.001")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="yolo26l_rgbpcar_rgb_tiff_scratch")
    parser.add_argument("--weights", type=Path, help="Optional initialization weights; default is random initialization")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from last.pt")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.weights and args.resume:
        raise ValueError("Use either --weights for a fresh run or --resume for checkpoint recovery, not both")
    dataset = args.dataset.expanduser().resolve()
    validate_split(dataset, "train")
    validate_split(dataset, "test")
    model_path, data_path = write_configs(dataset)

    model = YOLO(str(args.resume.resolve() if args.resume else model_path))
    if args.weights:
        model.load(str(args.weights.expanduser().resolve()))
    print(f"Model config: {model_path} (scale=l, RGB=3, nc=1)")
    print(f"Dataset config: {data_path} (original train/test RGB TIFF)")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=bool(args.weights) if not args.resume else False,
        resume=str(args.resume.resolve()) if args.resume else False,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.9,
        weight_decay=1e-4,
        cos_lr=True,
        warmup_bias_lr=0.0,
        amp=args.amp,
        deterministic=False,
        seed=args.seed,
        patience=0,
        val=True,
        save=True,
        plots=True,
        cache=False,
    )


if __name__ == "__main__":
    main(parse_args())
