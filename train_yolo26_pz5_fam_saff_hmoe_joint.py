"""Train dual RGB/DoLP FAM, sparse SAFF and depth HMoE Joint on P2-P5 from scratch."""

from train_yolo26_pz5 import ROOT, get_parser, main
from train_yolo26_pz5_pim_saff_hmoe_joint import PZ5PIMSAFFHMoETrainer
from ultralytics.nn.modules.pz import PZDualFAMStage


class PZ5FAMSAFFHMoETrainer(PZ5PIMSAFFHMoETrainer):
    """Reuse five-channel preprocessing and topology checks with FAM backbone stages."""

    stage_type = PZDualFAMStage


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz5-fam-saff-hmoe-joint.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml",
        workers=2,
        name="yolo26_pz5_fam_saff_hmoe_joint_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args, trainer_cls=PZ5FAMSAFFHMoETrainer)
