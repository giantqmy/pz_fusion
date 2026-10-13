"""Train dual RGB/DoLP PIM, sparse SAFF and depth HMoE Joint on P2-P5 from scratch."""

from train_yolo26_pz5 import ROOT, PZ5Trainer, get_parser, main
from ultralytics.nn.modules.pz import PZDualPIMStage, PZHMoEJointFusion, PZSAFFLateral


class PZ5PIMSAFFHMoETrainer(PZ5Trainer):
    """Use five-channel modality normalization and require the dual-backbone fusion graph."""

    stage_type = PZDualPIMStage

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        stages = (1, 6, 10, 14)
        laterals = (2, 7, 11, 15)
        fusions = (5, 9, 13, 17)
        for module, indices in ((self.stage_type, stages), (PZSAFFLateral, laterals), (PZHMoEJointFusion, fusions)):
            if tuple(i for i, layer in enumerate(model.model) if isinstance(layer, module)) != indices:
                raise ValueError(f"Expected {module.__name__} at layers {indices}")
        if tuple(model.model[i].f for i in stages) != (0, 1, 6, 10):
            raise ValueError("The RGB/DoLP backbone must continue from the preceding dual state")
        if any(model.model[i].f != stage for i, stage in zip(laterals, stages)):
            raise ValueError("Each SAFF lateral must consume its matching dual state")
        if tuple(model.model[i].f for i in fusions) != ([2, 4], [7, 8], [11, 12], [15, 16]):
            raise ValueError("HMoE Joint must fuse each SAFF lateral with its matching depth feature")
        if model.model[-1].f != [26, 29, 32, 35]:
            raise ValueError("The detector must consume P2-P5 PAN features")
        return model


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz5-pim-saff-hmoe-joint.yaml",
        data=ROOT / "ultralytics/cfg/datasets/pz6-rgb-dolp-depth.yaml",
        workers=2,
        name="yolo26_pz5_pim_saff_hmoe_joint_scratch",
    )
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("--model must be a YAML configuration for training from scratch")
    main(args, trainer_cls=PZ5PIMSAFFHMoETrainer)
