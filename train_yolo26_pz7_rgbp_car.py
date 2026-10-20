"""Train the RGB/color-DoLP PZ7 dual-wavelet model on prepared RGB-P Car data."""

import os
from pathlib import Path

from train_yolo26_pz7_pim_saffconcat_hmoe_joint_no_depth import PZ7NoDepthTrainer
from train_yolo26_pz5 import get_parser, main

ROOT = Path(__file__).resolve().parent
DATASET_CANDIDATES = (
    Path("/media/ddc/新加卷/hys/qmy/pz/RGB-P Car"),
    Path(r"I:\KEDADATA\rgb-p-car\RGB-P Car"),
)


def validate_prepared_dataset(dataset):
    """Validate existing paired YOLO data without creating or rewriting files."""
    train_list = dataset / "train.txt"
    test_list = dataset / "test.txt"
    dolp_root = dataset / "images" / "rgb_dolp"
    if not train_list.is_file() or not test_list.is_file() or not dolp_root.is_dir():
        raise FileNotFoundError(
            f"Prepared RGB-P Car data is incomplete under {dataset}; expected train.txt, test.txt, "
            "and images/rgb_dolp/. No data was generated."
        )

    entries = [line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"Prepared training list is empty: {train_list}")
    for entry in entries:
        image = Path(entry)
        if not image.is_absolute():
            image = dataset / image
        rgb = image.with_name(f"{image.stem}_S0_rgb.png")
        dolp = dolp_root / f"{image.stem}_dolp_color_rgb.png"
        image_parts = list(image.parts)
        if "images" not in image_parts:
            raise ValueError(f"Prepared image path must contain an images directory: {image}")
        image_parts[image_parts.index("images")] = "labels"
        label = Path(*image_parts).with_suffix(".txt")
        if not image.is_file() or not rgb.is_file() or not dolp.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing prepared RGB/DoLP image or label: {image}, {rgb}, {dolp}, {label}")


def resolve_dataset():
    """Find a previously prepared dataset; never fall back to raw source data."""
    configured = os.environ.get("RGBP_CAR_DATASET")
    candidates = (Path(configured),) if configured else DATASET_CANDIDATES
    for path in candidates:
        if (path / "train.txt").is_file() and (path / "test.txt").is_file():
            validate_prepared_dataset(path)
            return path
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Prepared RGB-P Car files were not found at: {locations}. "
        "Set RGBP_CAR_DATASET to the prepared dataset root. Raw data will not be converted."
    )


if __name__ == "__main__":
    dataset = resolve_dataset()
    parser = get_parser()
    model_path = ROOT / "ultralytics/cfg/models/26/yolo26-pz7-pim-saffconcat-no-depth.yaml"
    parser.set_defaults(
        model=model_path,
        data=ROOT / "ultralytics/cfg/datasets/rgb-p-car-rgb-dolp.yaml",
        workers=2,
        name="yolo26_pz7_rgbp_car_dualwt",
        val=False,
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    import yaml

    model_config = yaml.safe_load(args.model.read_text(encoding="utf-8"))
    model_config["nc"] = 1
    for layer in model_config["backbone"]:
        if layer[2] == "PZDualPIMStage":
            layer[2] = "PZDualWTStage"
    dataset_model = ROOT / "runs/rgbp_car/yolo26-pz7-rgbp-car-dualwt.yaml"
    dataset_model.parent.mkdir(parents=True, exist_ok=True)
    dataset_model.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
    args.model = dataset_model

    dataset_yaml = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_yaml["path"] = str(dataset)
    dataset_yaml["dolp_root"] = str(dataset / "images" / "rgb_dolp")
    dataset_config = ROOT / "runs/rgbp_car/rgb-p-car-rgb-dolp-prepared.yaml"
    dataset_config.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    args.data = dataset_config
    main(args, trainer_cls=PZ7NoDepthTrainer)
