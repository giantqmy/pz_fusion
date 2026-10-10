"""Fine-tune PZ5 with HMoE at P3; freeze the PZ4 visual backbone and train depth/fusion/neck/head."""

from train_yolo26_pz5_finetune import ROOT, PZ5FinetuneTrainer, get_parser, main
from ultralytics.nn.modules.pz import PZDepthFusion, PZHMoEFusion


class PZ5HMoEFinetuneTrainer(PZ5FinetuneTrainer):
    """Reuse strict PZ4 transfer and visual freezing with HMoE only at layer 8 (P3)."""

    fusion_types = (PZDepthFusion, PZHMoEFusion)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if [i for i, m in enumerate(model.model) if isinstance(m, PZHMoEFusion)] != [8]:
            raise ValueError("The HMoE experiment must replace only the P3 fusion at layer 8")
        return model


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz5-hmoe-p3.yaml",
        name="yolo26_pz5_hmoe_p3_from_pz4",
    )
    main(parser.parse_args(), trainer_cls=PZ5HMoEFinetuneTrainer)
