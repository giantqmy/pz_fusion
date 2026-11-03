"""Train reduced cross-query DualWT SAFF on original RGB-P Car TIFF pairs."""

from pathlib import Path

import yaml

from train_yolo26_pz5 import PZ5Trainer, get_parser, main
from train_yolo26_pz7_rgbp_car_dualwt import RGBPDualWTTrainer, resolve_dataset
from ultralytics.nn.modules.pz import (
    PZDepthFusion,
    PZDepthStage,
    PZDualWTStage,
    PZHMoEFusion,
    PZSAFFCrossQueryReducedLateral,
)

ROOT = Path(__file__).resolve().parent


class RGBPDualWTCrossQueryReducedTrainer(RGBPDualWTTrainer):
    """Validate the reduced cross-query SAFF topology before training."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PZ5Trainer.get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        stages = (1, 3, 5, 7)
        laterals = (2, 4, 6, 8)
        if tuple(i for i, layer in enumerate(model.model) if isinstance(layer, PZDualWTStage)) != stages:
            raise ValueError(f"Expected PZDualWTStage modules at layers {stages}")
        if tuple(i for i, layer in enumerate(model.model) if type(layer) is PZSAFFCrossQueryReducedLateral) != laterals:
            raise ValueError(f"Expected reduced cross-query SAFF modules at layers {laterals}")
        if any(isinstance(layer, (PZDepthStage, PZDepthFusion, PZHMoEFusion)) for layer in model.model):
            raise ValueError("The no-depth reduced cross-query model must not contain depth or HMoE modules")
        if tuple(model.model[i].f for i in stages) != (0, 1, 3, 5):
            raise ValueError("The dual visual backbone must continue through P2-P5")
        if any(model.model[i].f != stage for i, stage in zip(laterals, stages)):
            raise ValueError("Each reduced cross-query lateral must consume its matching dual state")
        if model.model[-1].f != [17, 20, 23, 26]:
            raise ValueError("The detector must consume the P2-P5 PAN features")
        return model


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--dataset", type=Path, default=None, help="RGB-P Car dataset root")
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26l-pz7-dualwt-saff-crossquery-reduced-no-depth.yaml",
        workers=2,
        name="yolo26l_pz7_rgbp_car_dualwt_saff_crossquery_reduced",
        val=True,
    )
    args = parser.parse_args()
    dataset = resolve_dataset(args.dataset)
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")

    dataset_config = ROOT / "runs/rgbp_car/rgb-p-car-rgb-dolp-tiff.yaml"
    dataset_config.parent.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "train/images",
                "val": "test/images",
                "channels": 6,
                "ch": 6,
                "rgb_dolp_tiff": True,
                "dolp_channels": 3,
                "names": {0: "Car"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    args.data = dataset_config
    main(args, trainer_cls=RGBPDualWTCrossQueryReducedTrainer)
