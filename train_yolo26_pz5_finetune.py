"""Initialize PZ5 from a trained RGB+DoLP PZ4 model and freeze its visual backbone.

Example:
    python train_yolo26_pz5_finetune.py --weights runs/train/yolo26_pz4/weights/best.pt \
        --data ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml --device 0

Depth stages and gates are newly initialized; neck and Detect weights are fine-tuned.
The model scale is inherited from the checkpoint. Visual BatchNorm statistics stay frozen.
"""

import argparse
from copy import deepcopy
from pathlib import Path

from train_yolo26_pz5 import PZ5Trainer
from ultralytics.nn.modules.pz import PZDepthFusion, PZDepthStage, PZPIMStage
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import LOGGER, YAML

ROOT = Path(__file__).resolve().parent
VISUAL_LAYERS = [1, 2, 6, 9, 12]
DEPTH_LAYERS = [3, 4, 7, 10, 13]
FUSION_LAYERS = [5, 8, 11, 14]


class PZ5FinetuneTrainer(PZ5Trainer):
    """Transfer PZ4 weights by semantic layer position instead of state_dict layer number."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None or weights.yaml.get("channels") != 4:
            raise ValueError("--weights must be a trained four-channel RGB+DoLP yolo26-pz4 checkpoint")
        if len(weights.yaml["backbone"]) != 5 or not all(isinstance(m, PZPIMStage) for m in weights.model[:5]):
            raise ValueError("Expected the five-stage PZ4 visual backbone, without depth stages or gates")
        source_names = [str(weights.names[i]) for i in range(weights.yaml["nc"])]
        target_names = [str(self.data["names"][i]) for i in range(self.data["nc"])]
        if source_names != target_names:
            raise ValueError(f"Checkpoint and dataset class order must match: {source_names} != {target_names}")

        cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        if cfg.get("channels") != 5 or len(cfg["backbone"]) != 15:
            raise ValueError("--model must use the yolo26-pz5 backbone (five-channel input, 15 layers)")
        # A filename like yolo26-pz5.yaml does not select the trained PZ4 model's scale.
        cfg["scale"] = weights.yaml.get("scale") or next(iter(weights.yaml["scales"]))
        self.data["channels"] = 5
        self.data["rgb_dolp_depth"] = True
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        if [i for i, m in enumerate(model.model[:15]) if isinstance(m, PZPIMStage)] != VISUAL_LAYERS:
            raise ValueError("Unexpected PZ5 visual layer positions")
        if [i for i, m in enumerate(model.model[:15]) if isinstance(m, PZDepthStage)] != DEPTH_LAYERS:
            raise ValueError("Unexpected PZ5 depth layer positions")
        if [i for i, m in enumerate(model.model[:15]) if isinstance(m, PZDepthFusion)] != FUSION_LAYERS:
            raise ValueError("Unexpected PZ5 fusion layer positions")
        if len(weights.model) - 5 != len(model.model) - 15:
            raise ValueError("PZ4 and PZ5 neck/head layer counts must match")
        if model.end2end != weights.end2end:
            raise ValueError("PZ4 and PZ5 must use the same end-to-end Detect configuration")

        mapping = list(zip(range(5), VISUAL_LAYERS)) + list(zip(range(5, len(weights.model)), range(15, len(model.model))))
        source = weights.float()
        # Validate every layer before copying: silent partial loading would invalidate the experiment.
        for old, new in mapping:
            src, dst = source.model[old], model.model[new]
            src_state, dst_state = src.state_dict(), dst.state_dict()
            if type(src) is not type(dst) or src_state.keys() != dst_state.keys():
                raise ValueError(f"Layer structure mismatch: PZ4[{old}] -> PZ5[{new}]")
            mismatches = [key for key in src_state if src_state[key].shape != dst_state[key].shape]
            if mismatches:
                raise ValueError(f"Layer size mismatch: PZ4[{old}] -> PZ5[{new}], keys: {mismatches}")
        for old, new in mapping:
            model.model[new].load_state_dict(source.model[old].state_dict(), strict=True)
            LOGGER.info(f"Loaded PZ4[{old}] -> PZ5[{new}] ({type(model.model[new]).__name__})")

        # BaseTrainer also puts these layers' BatchNorm modules in eval mode each epoch.
        # Do not use no_grad(): depth gradients must pass through the frozen visual stages.
        self.args.freeze = VISUAL_LAYERS.copy()
        YAML.save(
            self.save_dir / "transfer.yaml",
            {
                "pretrained": str(self.args.pretrained),
                "scale": cfg["scale"],
                "class_names": target_names,
                "layer_mapping": [{"pz4": old, "pz5": new} for old, new in mapping],
                "frozen_visual_layers": VISUAL_LAYERS,
                "new_depth_layers": DEPTH_LAYERS,
                "new_fusion_layers": FUSION_LAYERS,
                "finetuned_neck_and_head_layers": list(range(15, len(model.model))),
            },
        )
        LOGGER.info("Freeze visual backbone and its BN statistics; train depth/gates and fine-tune neck/Detect")
        return model


def main(args):
    for path in (args.weights, args.model, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    trainer = PZ5FinetuneTrainer(
        overrides={
            "task": "detect",
            "mode": "train",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "pretrained": str(args.weights.resolve()),
            "freeze": VISUAL_LAYERS.copy(),
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "optimizer": "AdamW",
            "lr0": args.lr0,
            "warmup_bias_lr": 0.0,
            "project": str(args.project.resolve()),
            "name": args.name,
        }
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="trained four-channel RGB+DoLP PZ4 checkpoint")
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/26/yolo26-pz5.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=1e-4, help="AdamW learning rate for all trainable layers")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="yolo26_pz5_from_pz4")
    main(parser.parse_args())
