"""Create train/test image manifests for RGB-P Car S0 RGB PNG images."""

import argparse
from pathlib import Path


DATASET_ROOT = Path("/media/ddc/\u65b0\u52a0\u5377/hys/qmy/pz/RGB-P Car")


def collect_images(directory: Path) -> list[Path]:
    images = sorted(path.resolve() for path in directory.rglob("*_S0_rgb.png") if path.is_file())
    if not images:
        raise FileNotFoundError(f"No *_S0_rgb.png images found under: {directory}")
    return images


def write_manifest(images: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(path.as_posix() for path in images) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} image paths to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()

    train_dir = args.root / "images/rgb_aolp_dolp/train"
    test_dir = args.root / "images/rgb_aolp_dolp/test"
    train_images = collect_images(train_dir)
    test_images = collect_images(test_dir)
    overlap = set(train_images) & set(test_images)
    if overlap:
        raise ValueError(f"Train/test image overlap detected, for example: {next(iter(overlap))}")

    write_manifest(train_images, args.root / "train_s0_rgb.txt")
    write_manifest(test_images, args.root / "test_s0_rgb.txt")


if __name__ == "__main__":
    main()
