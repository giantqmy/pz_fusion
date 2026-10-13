"""Train ConvNeXt dual features, wavelet PIM, sparse SAFF and depth HMoE Joint from scratch."""

from train_yolo26_pz5 import ROOT, get_parser, main
from train_yolo26_pz5_pim_saff_hmoe_joint import PZ5PIMSAFFHMoETrainer
from ultralytics.nn.modules.pz import PZConvNeXtPIMStage


class PZ5ConvNeXtPIMSAFFHMoETrainer(PZ5PIMSAFFHMoETrainer):
    """Use ConvNeXt feature extraction with the existing five-channel fusion pipeline."""

    stage_type = PZConvNeXtPIMStage


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz5-convnext-pim-saff-hmoe-joint.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml",
        workers=2,
        name="yolo26_pz5_convnext_pim_saff_hmoe_joint_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args, trainer_cls=PZ5ConvNeXtPIMSAFFHMoETrainer)
