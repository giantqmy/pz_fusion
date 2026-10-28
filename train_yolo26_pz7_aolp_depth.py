"""Train the PZ7 RGB + color AoLP + depth model from scratch."""

from pathlib import Path

from train_yolo26_pz5 import ROOT, PZ5Trainer, get_parser, main
from ultralytics.models.yolo.detect import DetectionTrainer


class PZ7AoLPDepthTrainer(PZ5Trainer):
    """Train with RGB, three color AoLP channels and normalized depth input."""

    def build_dataset(self, img_path, mode="train", batch=None):
        self.data["channels"] = 7
        self.data["rgb_aolp_depth"] = True
        self.data.pop("rgb_dolp_depth", None)
        self.data.pop("rgb_dolp", None)
        self.data.pop("rgb_aolp_dolp", None)
        return DetectionTrainer.build_dataset(self, img_path, mode, batch)


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz7-pim-saffconcat-hmoe-joint.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz7-rgb-color-aolp-depth.yaml",
        workers=2,
        name="yolo26l_pz7_pim_saffconcat_hmoe_joint_aolp_depth_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args, trainer_cls=PZ7AoLPDepthTrainer)
