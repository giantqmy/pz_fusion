"""Train the bundled PCDNet model on original RGB-P Car TIFF triplets."""

import csv
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp

from pcdnet import load_checkpoint
from pcdnet.models.yolo import IDetect, Model
from pcdnet.utils.general import box_iou, non_max_suppression, xywh2xyxy
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ap_per_class

from .loss import ComputePCDNetLoss


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _reset_inference_grids(model):
    detect = _unwrap(model).model[-1]
    detect.grid = [torch.empty(0, device=detect.anchors.device) for _ in range(detect.nl)]


def split_polarization_inputs(images):
    """Split RGB-first BCHW data into PCDNet's RGB, AoLP and DoLP domains."""
    if images.ndim != 4 or images.shape[1] != 9:
        raise ValueError(f"Expected RGB+AoLP+DoLP BCHW input with 9 channels, got {tuple(images.shape)}")
    return [images[:, :3], images[:, 3:6], images[:, 6:9]]


def _batch_targets(batch, device):
    return torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(device)


def _class_names(names):
    if isinstance(names, dict):
        return [str(names[index]) for index in range(len(names))]
    return [str(name) for name in names]


def _reset_detection_head(model, nc, class_names):
    detect = model.model[-1]
    if not isinstance(detect, IDetect):
        raise TypeError(f"PCDNet checkpoint ends in {type(detect).__name__}, expected IDetect")
    old_nc, old_no = detect.nc, detect.no
    if old_nc != nc:
        new_no = nc + 5
        new_convs, new_implicit = nn.ModuleList(), nn.ModuleList()
        for old_conv, old_implicit in zip(detect.m, detect.im):
            new_conv = nn.Conv2d(old_conv.in_channels, detect.na * new_no, 1, bias=True)
            with torch.no_grad():
                old_w = old_conv.weight.view(detect.na, old_no, *old_conv.weight.shape[1:])
                old_b = old_conv.bias.view(detect.na, old_no)
                new_w = new_conv.weight.view(detect.na, new_no, *new_conv.weight.shape[1:])
                new_b = new_conv.bias.view(detect.na, new_no)
                new_w[:, :5].copy_(old_w[:, :5])
                new_b[:, :5].copy_(old_b[:, :5])
                common = min(old_nc, nc)
                if common:
                    new_w[:, 5 : 5 + common].copy_(old_w[:, 5 : 5 + common])
                    new_b[:, 5 : 5 + common].copy_(old_b[:, 5 : 5 + common])
            new_convs.append(new_conv)
            new_implicit.append(type(old_implicit)(detect.na * new_no))
        detect.m, detect.im = new_convs, new_implicit
        detect.nc, detect.no = nc, new_no
        model.yaml["nc"] = nc
    model.names = class_names
    return old_nc != nc


def _load_model(checkpoint_path, device, nc, names, pretrained=False, resume=False):
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a PCDNet checkpoint dictionary containing model or ema")
    template = next(
        (checkpoint.get(key) for key in ("ema", "model") if isinstance(checkpoint.get(key), nn.Module)), None
    )
    if template is None:
        raise ValueError("Training requires the author's full-object PCDNet.pt, not a state_dict-only checkpoint")
    template = _unwrap(template)
    if resume or pretrained:
        model = template.float()
    else:
        if not hasattr(template, "yaml"):
            raise ValueError("The architecture checkpoint does not contain PCDNet's embedded model YAML")
        model = Model(cfg=deepcopy(template.yaml), nc=nc).float()
    changed = _reset_detection_head(model, nc, _class_names(names))
    return model, checkpoint if resume else {}, changed


def _make_loader(data, cfg, split, batch_size, workers, augment):
    mode = "train" if augment else "val"
    dataset = build_yolo_dataset(cfg, data[split], batch_size, data, mode=mode, rect=False, stride=32)
    return build_dataloader(dataset, batch_size, workers, shuffle=split == "train", rank=-1)


def _match_predictions(predictions, labels, iouv):
    correct = torch.zeros((predictions.shape[0], iouv.numel()), dtype=torch.bool, device=predictions.device)
    if not len(labels) or not len(predictions):
        return correct
    iou = box_iou(labels[:, 1:], predictions[:, :4])
    matches = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == predictions[:, 5]))
    if matches[0].numel():
        match_data = torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1).cpu().numpy()
        if match_data.shape[0] > 1:
            match_data = match_data[match_data[:, 2].argsort()[::-1]]
            match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
            match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]
        match_data = torch.as_tensor(match_data, device=predictions.device)
        correct[match_data[:, 1].long()] = match_data[:, 2:3] >= iouv
    return correct


class PCDNetTrainer:
    """Train PCDNet from scratch, from released parameters, or from a resumable checkpoint."""

    def __init__(self, args):
        self.args = args
        self.device = self._select_device(args.device)
        self._seed_everything(args.seed)
        self.data = check_det_dataset(args.data, autodownload=False)
        if self.data.get("channels") != 9 or not self.data.get("rgb_aolp_dolp_tiff"):
            raise ValueError("PCDNet training data must set channels: 9 and rgb_aolp_dolp_tiff: true")
        self.names, self.nc = self.data["names"], self.data["nc"]
        self.save_dir = Path(args.project) / args.name
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        source_checkpoint = args.resume or args.weights
        if not Path(source_checkpoint).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {source_checkpoint}. PCDNet.pt is required because the author did not publish YAML."
            )
        self.model, self.checkpoint, head_changed = _load_model(
            source_checkpoint,
            self.device,
            self.nc,
            self.names,
            pretrained=args.pretrained,
            resume=bool(args.resume),
        )
        self.initialization = "resume" if args.resume else "released-weights" if args.pretrained else "random"
        print(f"PCDNet initialization: {self.initialization} (seed={args.seed})")
        self.model.to(self.device)
        _reset_inference_grids(self.model)
        if "," in args.device and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        overrides = {
            "task": "detect",
            "imgsz": args.imgsz,
            "cache": args.cache,
            "single_cls": False,
            "rect": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": args.mosaic if args.augment else 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
        }
        cfg = get_cfg(overrides=overrides)
        self.train_loader = _make_loader(self.data, cfg, "train", args.batch, args.workers, args.augment)
        self.val_loader = _make_loader(self.data, cfg, "val", args.batch, args.workers, False)
        self.criterion = ComputePCDNetLoss(
            _unwrap(self.model), box=args.box, obj=args.obj, cls=args.cls, anchor_t=args.anchor_t
        )

        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
        self.optimizer = torch.optim.SGD(
            no_decay, lr=args.lr0, momentum=args.momentum, weight_decay=0.0, nesterov=True
        )
        self.optimizer.add_param_group({"params": decay, "weight_decay": args.weight_decay})
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: args.lrf
            + (1 - args.lrf) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2,
        )
        self.scaler = amp.GradScaler(enabled=args.amp and self.device.type == "cuda")
        self.start_epoch, self.best_map = 0, -1.0
        if args.resume:
            if head_changed:
                raise ValueError("Cannot resume after changing the detection head class count")
            self._restore_training_state()

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _select_device(spec):
        if spec.lower() == "cpu":
            return torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = spec
        if not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda:0")

    def _restore_training_state(self):
        if self.checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        if self.checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(self.checkpoint["scheduler"])
        self.start_epoch = int(self.checkpoint.get("epoch", -1)) + 1
        self.best_map = float(self.checkpoint.get("best_map", -1.0))

    def train(self):
        csv_path = self.save_dir / "results.csv"
        if self.start_epoch == 0:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    ["epoch", "train_box", "train_obj", "train_cls", "val_loss", "precision", "recall", "map50", "map50_95"]
                )
        for epoch in range(self.start_epoch, self.args.epochs):
            components = self._train_epoch()
            metrics = self.validate()
            self.scheduler.step()
            row = [
                epoch,
                *components,
                metrics["loss"],
                metrics["precision"],
                metrics["recall"],
                metrics["map50"],
                metrics["map"],
            ]
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow([f"{value:.6g}" if isinstance(value, float) else value for value in row])
            is_best = metrics["map"] > self.best_map
            self.best_map = max(self.best_map, metrics["map"])
            self._save(epoch, self.weights_dir / "last.pt")
            if is_best:
                self._save(epoch, self.weights_dir / "best.pt")
            print(
                f"epoch {epoch + 1}/{self.args.epochs} | box {components[0]:.4f} obj {components[1]:.4f} "
                f"cls {components[2]:.4f} | P {metrics['precision']:.4f} R {metrics['recall']:.4f} "
                f"mAP50 {metrics['map50']:.4f} mAP50-95 {metrics['map']:.4f}"
            )

    def _train_epoch(self):
        self.model.train()
        running = torch.zeros(3, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            images = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            targets = _batch_targets(batch, self.device)
            with amp.autocast(enabled=self.scaler.is_enabled()):
                predictions = self.model(split_polarization_inputs(images))
                loss, parts = self.criterion(predictions, targets)
                loss = loss / self.args.accumulate
            self.scaler.scale(loss).backward()
            if (step + 1) % self.args.accumulate == 0 or step + 1 == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            running = (running * step + parts) / (step + 1)
        return tuple(float(value) for value in running.cpu())

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        stats, loss_sum = [], 0.0
        for batch in self.val_loader:
            images = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            targets = _batch_targets(batch, self.device)
            predictions, raw = self.model(split_polarization_inputs(images))
            loss_sum += float(self.criterion(raw, targets)[0]) / images.shape[0]
            detections = non_max_suppression(predictions, conf_thres=0.001, iou_thres=0.65, multi_label=True)
            height, width = images.shape[2:]
            for image_index, detection in enumerate(detections):
                labels = targets[targets[:, 0] == image_index, 1:].clone()
                target_classes = labels[:, 0].tolist()
                if len(labels):
                    labels[:, 1:] *= torch.tensor([width, height, width, height], device=self.device)
                    labels[:, 1:] = xywh2xyxy(labels[:, 1:])
                correct = _match_predictions(detection, labels, iouv)
                stats.append((correct.cpu(), detection[:, 4].cpu(), detection[:, 5].cpu(), target_classes))

        precision = recall = map50 = mean_ap = 0.0
        if stats:
            combined = [np.concatenate(values, 0) for values in zip(*stats)]
            if combined[0].shape[0] and combined[3].shape[0]:
                result = ap_per_class(*combined, names={index: name for index, name in enumerate(_class_names(self.names))})
                precision, recall, ap = result[2], result[3], result[5]
                map50, mean_ap = float(ap[:, 0].mean()), float(ap.mean())
                precision, recall = float(precision.mean()), float(recall.mean())
        return {
            "loss": loss_sum / max(len(self.val_loader), 1),
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map": mean_ap,
        }

    def _save(self, epoch, path):
        checkpoint = {
            "epoch": epoch,
            "best_map": self.best_map,
            "model": deepcopy(_unwrap(self.model)).half(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "data": str(self.args.data),
            "channel_mapping": "RGB TIFF + AoLP TIFF + DoLP TIFF",
            "initialization": self.initialization,
            "train_args": vars(self.args),
        }
        torch.save(checkpoint, path)
