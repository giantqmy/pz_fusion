"""Focused CPU validation of HMoE transfer, gradients and frozen visual BatchNorm."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from train_yolo26_pz5_finetune import VISUAL_LAYERS, DEPTH_LAYERS, FUSION_LAYERS, PZ5FinetuneTrainer
from train_yolo26_pz5_hmoe_finetune import PZ5HMoEFinetuneTrainer
from ultralytics.nn.modules.pz import PZHMoEFusion, PZDepthFusion
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

torch.set_num_threads(1)
cfg = yaml_model_load('ultralytics/cfg/models/26/yolo26-pz5.yaml')
# Build a synthetic five-stage PZ4 baseline without requiring an external checkpoint/config.
cfg['channels'] = cfg['ch'] = 4
cfg['backbone'] = [[-1, 1, 'PZPIMStage', args] for _, _, _, args in [cfg['backbone'][i] for i in VISUAL_LAYERS]]
old_to_new = {5: 1, 8: 2, 11: 3, 14: 4, **{i: i - 10 for i in range(15, 28)}}
for entry in cfg['head']:
    entry[0] = [old_to_new.get(i, i) for i in entry[0]] if isinstance(entry[0], list) else old_to_new.get(entry[0], entry[0])
cfg['scale'] = 'n'
source = DetectionModel(cfg, ch=4, nc=6, verbose=False)
source.names = {i: str(i) for i in range(6)}
with TemporaryDirectory(dir='.') as temp:
    trainer = object.__new__(PZ5HMoEFinetuneTrainer)
    trainer.data = {'nc': 6, 'names': source.names}
    trainer.args = SimpleNamespace(pretrained='synthetic-pz4.pt')
    trainer.save_dir = Path(temp)
    model = trainer.get_model('ultralytics/cfg/models/26/yolo26-pz5-hmoe-p3.yaml', source, verbose=False)
    assert trainer.args.freeze == VISUAL_LAYERS
    assert isinstance(model.model[8], PZHMoEFusion)
    assert all(isinstance(model.model[i], PZDepthFusion) for i in [5, 11, 14])
    for old, new in zip(range(5), VISUAL_LAYERS):
        assert all(torch.equal(v, model.model[new].state_dict()[k]) for k, v in source.model[old].state_dict().items())
    for old, new in zip(range(5, len(source.model)), range(15, len(model.model))):
        assert all(torch.equal(v, model.model[new].state_dict()[k]) for k, v in source.model[old].state_dict().items())
    model.train()
    for i in VISUAL_LAYERS:
        model.model[i].requires_grad_(False)
        for layer in model.model[i].modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
    bn = [m for i in VISUAL_LAYERS for m in model.model[i].modules() if isinstance(m, torch.nn.BatchNorm2d)]
    running = [m.running_mean.clone() for m in bn]
    outputs = []
    handles = [model.model[i].register_forward_hook(lambda m, a, out: outputs.append(out)) for i in FUSION_LAYERS]
    model(torch.randn(2, 5, 64, 64))
    sum(x.square().mean() for x in outputs).backward()
    assert all(p.grad is None for i in VISUAL_LAYERS for p in model.model[i].parameters())
    assert all(torch.equal(old, m.running_mean) for old, m in zip(running, bn))
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[i].parameters()) for i in DEPTH_LAYERS)
    for name in ['gate_thi', 'expert_a', 'expert_b', 'dispatch_temp', 'combine_temp', 'alpha']:
        grad = getattr(model.model[8], name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    for handle in handles:
        handle.remove()
    original = object.__new__(PZ5FinetuneTrainer)
    original.data, original.args, original.save_dir = trainer.data, trainer.args, trainer.save_dir
    baseline = original.get_model('ultralytics/cfg/models/26/yolo26-pz5.yaml', source, verbose=False)
    assert all(isinstance(baseline.model[i], PZDepthFusion) for i in FUSION_LAYERS)
print('PASS: P3-only HMoE, strict visual/neck/head transfer, frozen visual gradients/BN, depth and routing gradients, original trainer')
