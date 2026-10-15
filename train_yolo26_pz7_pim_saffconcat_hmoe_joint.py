"""Train RGB + three-channel DoLP + depth with PIM, SAFFConcat and HMoE Joint from scratch."""

from train_yolo26_pz5 import ROOT, get_parser, main
from train_yolo26_pz5_pim_saffconcat_hmoe_joint import PZ5PIMSAFFConcatHMoETrainer


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz7-pim-saffconcat-hmoe-joint.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz7-rgb-color-dolp-depth.yaml",
        workers=2,
        name="yolo26_pz7_pim_saffconcat_hmoe_joint_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args, trainer_cls=PZ5PIMSAFFConcatHMoETrainer)
