"""Infer one RGB + color DoLP + depth image and save its annotated PNG and YOLO TXT.

--source accepts a dataset base image (scene.png) or its RGB image
(scene_S0_rgb.png). Companions are scene_dolp_color_rgb.png and
scene_depth_dense_u16.png. DoLP defaults to beside the RGB image; depth defaults
to its parent directory's depth/ folder. --data supplies modality roots and
depth normalization, and --dolp-root/--depth-root override their directories.

TXT rows: class_id x_center y_center width height confidence (normalized xywh).
Every run replaces the two output files; no detections produces an empty TXT.
No ground-truth labels or dataset split lists are needed.

Example:
    python predict_pz7_image.py --weights /path/to/best.pt \
        --source /dataset/images/val/scene_S0_rgb.png \
        --data ultralytics/cfg/datasets/pz7-rgb-color-dolp-depth.yaml \
        --output runs/predict_pz7 --device 0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import select_device
from visualize_two_weights_paper import infer, load_modalities, model_channels, names_dict, resolve_path, save_image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", "--net", type=Path, required=True, help="Seven-channel model .pt checkpoint")
    parser.add_argument("--source", "--image", type=Path, required=True, help="Absolute path to one base or S0 RGB image")
    parser.add_argument("--data", type=Path, help="Dataset YAML for modality roots and training depth normalization")
    parser.add_argument("--dolp-root", type=Path, help="Directory containing *_dolp_color_rgb.png")
    parser.add_argument("--depth-root", type=Path, help="Directory containing *_depth_dense_u16.png")
    parser.add_argument("--output", type=Path, default=Path("runs/predict_pz7"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--box-width", type=int, default=3)
    return parser.parse_args()


def main(args):
    weights = args.weights.expanduser().resolve()
    source = args.source.expanduser().resolve()
    data_file = args.data.expanduser().resolve() if args.data else Path.cwd() / "data.yaml"
    for path in (weights, source, *([data_file] if args.data else [])):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.imgsz <= 0 or not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("imgsz must be positive; conf and iou must be between 0 and 1")
    data = YAML.load(data_file) if args.data else {}
    for key in ("dolp_root", "depth_root"):
        value = getattr(args, key) or data.get(key)
        if value:
            data[key] = str(resolve_path(value, data_file.parent))
    device = select_device(args.device)
    model = YOLO(str(weights))
    if model_channels(model) != 7:
        raise ValueError(f"Expected a 7-channel RGB + color DoLP + depth model, got {model_channels(model)}")
    model.model.to(device).eval()
    args.imgsz = check_imgsz(args.imgsz, stride=model.model.stride)
    # The shared loader derives companion paths from a base stem; a base file need not exist for an S0 input.
    base = source.with_name(f"{source.stem[:-7]}.png") if source.stem.endswith("_S0_rgb") else source
    image = load_modalities(base, data, 7, data_file, data.get("dolp_root"), data.get("depth_root"))
    detections = infer(model, image, data, args, device)
    rgb_bgr = cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    result = Results(rgb_bgr, path=str(source), names=names_dict(model.names), boxes=detections)
    output = args.output.expanduser().resolve()
    image_file, txt_file = output / f"{base.stem}.png", output / f"{base.stem}.txt"
    save_image(image_file, result.plot(conf=True, labels=True, line_width=args.box_width, color_mode="class"))
    # Results.save_txt appends by default; truncate first and also create a file for zero detections.
    txt_file.write_text("", encoding="utf-8")
    result.save_txt(str(txt_file), save_conf=True)
    print(f"Detections: {len(detections)}")
    print(f"Saved image: {image_file}")
    print(f"Saved TXT: {txt_file}")
    return result


if __name__ == "__main__":
    main(parse_args())
