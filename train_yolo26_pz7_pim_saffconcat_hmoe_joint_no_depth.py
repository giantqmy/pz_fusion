"""Train the RGB/color-DoLP PZ7 ablation without the depth branch."""

from train_yolo26_pz5 import ROOT, PZ5Trainer, get_parser, main
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.pz import (
    PZDepthFusion,
    PZDepthStage,
    PZDualWTStage,
    PZHMoEFusion,
    PZSAFFConcatLateral,
)


class PZ7NoDepthTrainer(PZ5Trainer):
    """Train a P2-P5 RGB/color-DoLP detector with no depth-specific modules."""

    def build_dataset(self, img_path, mode="train", batch=None):
        self.data["channels"] = 6
        self.data["rgb_dolp"] = True
        self.data.pop("rgb_dolp_depth", None)
        return DetectionTrainer.build_dataset(self, img_path, mode, batch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        stages = (1, 3, 5, 7)
        laterals = (2, 4, 6, 8)
        if tuple(i for i, layer in enumerate(model.model) if isinstance(layer, PZDualWTStage)) != stages:
            raise ValueError(f"Expected PZDualWTStage modules at layers {stages}")
        if tuple(i for i, layer in enumerate(model.model) if isinstance(layer, PZSAFFConcatLateral)) != laterals:
            raise ValueError(f"Expected PZSAFFConcatLateral modules at layers {laterals}")
        depth_modules = (PZDepthStage, PZDepthFusion, PZHMoEFusion)
        if any(isinstance(layer, depth_modules) for layer in model.model):
            raise ValueError("The no-depth ablation model must not contain depth or HMoE fusion modules")
        if tuple(model.model[i].f for i in stages) != (0, 1, 3, 5):
            raise ValueError("The dual visual backbone must continue through P2-P5")
        if any(model.model[i].f != stage for i, stage in zip(laterals, stages)):
            raise ValueError("Each SAFFConcat lateral must consume its matching dual visual state")
        if model.model[-1].f != [17, 20, 23, 26]:
            raise ValueError("The detector must consume the P2-P5 PAN features")
        return model


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz7-pim-saffconcat-no-depth.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz7-rgb-color-dolp.yaml",
        workers=2,
        name="yolo26_pz7_pim_saffconcat_no_depth_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args, trainer_cls=PZ7NoDepthTrainer)
