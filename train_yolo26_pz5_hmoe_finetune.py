"""Fine-tune PZ5 with HMoE at P2-P5; freeze the PZ4 visual backbone and train depth/fusion/neck/head."""

from train_yolo26_pz5_finetune import FUSION_LAYERS, ROOT, PZ5FinetuneTrainer, get_parser, main
from ultralytics.nn.modules.pz import PZHMoEFusion


class PZ5HMoEFinetuneTrainer(PZ5FinetuneTrainer):
    """Reuse strict PZ4 transfer and visual freezing with HMoE at all four fusion layers."""

    fusion_types = (PZHMoEFusion,)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if [i for i, m in enumerate(model.model) if isinstance(m, PZHMoEFusion)] != FUSION_LAYERS:
            raise ValueError("The HMoE experiment must replace all P2-P5 fusion layers")
        return model


if __name__ == "__main__":
    parser = get_parser()
    parser.set_defaults(
        model=ROOT / "ultralytics/cfg/models/26/yolo26-pz5-hmoe-all.yaml",
        name="yolo26_pz5_hmoe_all_from_pz4",
    )
    main(parser.parse_args(), trainer_cls=PZ5HMoEFinetuneTrainer)
