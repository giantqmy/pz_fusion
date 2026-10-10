"""Train P2-P5 HMoE with joint visual/depth output projection from random initialization."""

from train_yolo26_pz5 import ROOT, get_parser, main


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz5-hmoe-joint.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml",
        workers=2,
        name="yolo26_pz5_hmoe_joint_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args)
