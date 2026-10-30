"""Train YOLOv9c from scratch using RGB-only images."""

from pathlib import Path

from train_yolo26_pz5 import ROOT, get_parser
from ultralytics import YOLO
from ultralytics.utils import YAML


def main():
    """Train and validate the RGB-only YOLOv9c baseline."""
    parser = get_parser()
    parser.description = __doc__
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/v9/yolov9c.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz-rgb-only.yaml",
        workers=2,
        name="yolov9c_pz_rgb_scratch",
    )
    args = parser.parse_args()

    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration when training from scratch")
    data = YAML.load(args.data)
    if data.get("channels", 3) != 3 or any(
        data.get(key) for key in ("rgb_dolp", "rgb_dolp_depth", "rgb_aolp_dolp")
    ):
        parser.error("--data must describe RGB-only, three-channel input")

    model = YOLO(str(args.resume or args.model))
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        rect=args.rect,
        project=str(args.project),
        name=args.name,
        pretrained=False,
        freeze=0,
        resume=str(args.resume) if args.resume else False,
        val=getattr(args, "val", True),
        optimizer="auto",
    )


if __name__ == "__main__":
    main()
