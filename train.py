from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from contextlib import nullcontext
from torch.utils.data import DataLoader


LOG = logging.getLogger("train")

@dataclass
class TrainConfig:
    data_dir: Path
    resume: Optional[Path]
    epochs: int
    batch_size: int

    # Input
    img_size: int

    # Augmentation schedule
    disable_mix_last: int

    lr_head: float
    lr_backbone: float
    weight_decay: float

    unfreeze_epoch: int
    unfreeze: str  # "layer4" | "all"

    # Regularization / augmentation knobs
    label_smoothing: float
    mixup_alpha: float
    cutmix_alpha: float
    mix_prob: float

    # Scheduler / training control
    scheduler: str  # "plateau" | "cosine" | "none"
    early_stop_patience: int

    # Metrics / reporting
    topk: int
    write_metrics: bool
    best_metric: str  # "loss" | "acc"
    metrics_out: Path
    
    top_confusions: int

    # Runtime
    num_workers: int
    torch_threads: int
    seed: int
    device: str  # "auto" | "cpu" | "mps" | "cuda"
    debug: bool
    
    tta: bool = False  # eval-time test-time augmentation (horizontal flip)
    overwrite_metrics: bool = False
    run_name: str = ""



def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("vehicle-dataset"),
                   help="Dataset root containing train/ and test/")
    p.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224,
                   help="Input image size (train RandomResizedCrop and val CenterCrop). Common: 224, 320, 384")
    p.add_argument("--disable-mix-last", type=int, default=0,
                   help="Disable MixUp/CutMix for the last N epochs (0 = never disable)")
    p.add_argument("--tta", action="store_true", help="Enable simple eval-time TTA (horizontal flip)")
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", "--wd", type=float, default=1e-4)

    p.add_argument("--unfreeze-epoch", type=int, default=5, help="Epoch to start unfreezing backbone (0-based)")
    p.add_argument("--unfreeze", choices=["layer4", "all"], default="layer4",
                   help="Which part of backbone to unfreeze at unfreeze-epoch")

    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CrossEntropy label_smoothing in [0,1). Typical: 0.05 or 0.1")
    # Aliases for convenience/backwards-compat
    p.add_argument("--ls", type=float, default=None,
                   help="Alias for --label-smoothing")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Enable MixUp with Beta(alpha, alpha). Typical: 0.2")
    p.add_argument("--cutmix-alpha", type=float, default=0.0,
                   help="Enable CutMix with Beta(alpha, alpha). Typical: 1.0")
    p.add_argument("--mix-prob", type=float, default=1.0,
                   help="Probability of applying MixUp/CutMix when enabled (0-1)")
    p.add_argument("--aug", type=str, default=None,
                   help=("Augmentation preset. Examples: "
                         "'none' | 'mixup0.2@p0.6' | 'cutmix1.0@p0.6' | "
                         "'mixup0.2+cutmix1.0@p0.6'."))

    p.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau",
                   help="LR scheduler")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Early stop after N epochs without val_loss improvement (0 disables)")
    # Alias for convenience/backwards-compat
    p.add_argument("--es", type=int, default=None,
                   help="Alias for --early-stop-patience")

    p.add_argument("--topk", type=int, default=5, help="Compute Top-K accuracy (K>=1).")
    p.add_argument("--write-metrics", action="store_true",
                   help="Write confusion matrix + per-class metrics to --metrics-out at end.")

    p.add_argument("--best-metric", choices=["loss", "acc"], default="loss",
                   help="Which metric to use for selecting the best checkpoint and final metrics export.")
    p.add_argument("--metrics-out", type=Path, default=Path("checkpoints/metrics.json"))
    p.add_argument("--run-name", type=str, default="", help="Optional run name suffix for saved metrics/ckpts")
    p.add_argument("--overwrite-metrics", action="store_true", help="Allow overwriting --metrics-out if it already exists")
    p.add_argument("--top-confusions", type=int, default=10, help="Top off-diagonal confusions to report/write")

    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--torch-threads", type=int, default=0, help="If >0, set torch.set_num_threads()")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--debug", action="store_true", help="Enable verbose logging and print detailed config")
    a = p.parse_args()

    # Apply shorthand aliases if provided.
    if getattr(a, "ls", None) is not None:
        a.label_smoothing = a.ls
    if getattr(a, "es", None) is not None:
        a.early_stop_patience = a.es

    # Parse --aug preset (optional). This sets mixup/cutmix and probability in one go.
    if getattr(a, "aug", None) is not None:
        spec = str(a.aug).strip().lower()
        if spec in {"none", "off", "no", "0"}:
            a.mixup_alpha = 0.0
            a.cutmix_alpha = 0.0
            a.mix_prob = 0.0
        else:
            prob = 1.0
            base = spec
            if "@p" in spec:
                base, ppart = spec.split("@p", 1)
                try:
                    prob = float(ppart)
                except ValueError as e:
                    raise SystemExit(f"Invalid --aug probability in '{a.aug}'. Expected '@p<0-1>'.") from e
            mixup = 0.0
            cutmix = 0.0
            parts = [p for p in base.split("+") if p]
            for part in parts:
                if part.startswith("mixup"):
                    val = part.replace("mixup", "", 1)
                    mixup = float(val) if val else 0.2
                elif part.startswith("cutmix"):
                    val = part.replace("cutmix", "", 1)
                    cutmix = float(val) if val else 1.0
                else:
                    raise SystemExit(
                        f"Invalid --aug token '{part}' in '{a.aug}'. "
                        "Use 'none', 'mixup<alpha>', 'cutmix<alpha>', or 'mixup<alpha>+cutmix<alpha>' "
                        "optionally followed by '@p<prob>'."
                    )
            a.mixup_alpha = float(mixup)
            a.cutmix_alpha = float(cutmix)
            a.mix_prob = float(prob)

    return TrainConfig(
        data_dir=a.data_dir,
        resume=a.resume,
        epochs=a.epochs,
        batch_size=a.batch_size,
        img_size=a.img_size,
        disable_mix_last=a.disable_mix_last,
        lr_head=a.lr_head,
        lr_backbone=a.lr_backbone,
        weight_decay=a.weight_decay,
        unfreeze_epoch=a.unfreeze_epoch,
        unfreeze=a.unfreeze,
        label_smoothing=a.label_smoothing,
        mixup_alpha=a.mixup_alpha,
        cutmix_alpha=a.cutmix_alpha,
        mix_prob=a.mix_prob,
        scheduler=a.scheduler,
        early_stop_patience=a.early_stop_patience,
        topk=max(1, int(a.topk)),
        write_metrics=a.write_metrics,
        best_metric=a.best_metric,
        metrics_out=a.metrics_out,
        run_name=a.run_name,
        overwrite_metrics=a.overwrite_metrics,
        top_confusions=a.top_confusions,
        num_workers=a.num_workers,
        torch_threads=a.torch_threads,
        seed=a.seed,
        device=a.device,
        debug=a.debug,
        tta=a.tta,
    )


# ----------------------------
# Utilities
# ----------------------------

def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless if CUDA absent


def _build_transforms(img_size: int) -> Tuple[T.Compose, T.Compose]:
    # Match torchvision common convention: for 224 crop, resize to 256 (i.e., /0.875)
    val_resize = int(round(img_size / 0.875))
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(val_resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def _write_classmap(train_dataset: torchvision.datasets.ImageFolder, out_path: Path) -> Dict[int, str]:
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    out_path.write_text(json.dumps(idx_to_class, indent=2))
    return idx_to_class


def _set_trainable(model: nn.Module, *, mode: str) -> None:
    """
    mode:
      - "head": only fc trainable
      - "layer4": fc + layer4 trainable
      - "all": entire model trainable
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    if mode == "head":
        return

    if mode == "layer4":
        # ResNet has layer1..layer4
        for p in model.layer4.parameters():
            p.requires_grad = True
        return

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unknown mode: {mode}")


def _make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float) -> optim.Optimizer:
    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]

    # Note: backbone_params may be empty in the "head" phase.
    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def _print_config(cfg: TrainConfig,
                  device: torch.device,
                  pin_memory: bool,
                  use_amp: bool,
                  train_size: int,
                  val_size: int,
                  classes: Sequence[str],
                  start_epoch: int) -> None:
    mix_desc = "none"
    if cfg.mixup_alpha > 0 and cfg.cutmix_alpha > 0:
        mix_desc = f"mixup{cfg.mixup_alpha:g}+cutmix{cfg.cutmix_alpha:g}@p{cfg.mix_prob:g}"
    elif cfg.mixup_alpha > 0:
        mix_desc = f"mixup{cfg.mixup_alpha:g}@p{cfg.mix_prob:g}"
    elif cfg.cutmix_alpha > 0:
        mix_desc = f"cutmix{cfg.cutmix_alpha:g}@p{cfg.mix_prob:g}"

    LOG.info(
        "Config: device=%s pin_memory=%s amp=%s workers=%d batch=%d epochs=%d img=%d "
        "lr_head=%.3g lr_backbone=%.3g wd=%.3g unfreeze=%s@%d "
        "ls=%.3g aug=%s sch=%s es=%s disable_mix_last=%d tta=%s best=%s "
        "train=%d val=%d classes=%d resume=%s run=%s",
        device.type, pin_memory, use_amp, cfg.num_workers, cfg.batch_size, cfg.epochs, cfg.img_size,
        cfg.lr_head, cfg.lr_backbone, cfg.weight_decay, cfg.unfreeze, cfg.unfreeze_epoch,
        cfg.label_smoothing, mix_desc, cfg.scheduler,
        str(cfg.early_stop_patience) if cfg.early_stop_patience > 0 else "off",
        cfg.disable_mix_last, str(cfg.tta), cfg.best_metric,
        train_size, val_size, len(classes), str(cfg.resume) if cfg.resume else "none", cfg.run_name or "auto",
    )
    if not cfg.debug:
        return

    detail = {
        "platform": {"python": platform.python_version(), "os": platform.platform(), "machine": platform.machine()},
        "device": {"device": str(device), "pin_memory": pin_memory, "amp_enabled": use_amp},
        "data": {"data_dir": str(cfg.data_dir), "train_samples": train_size, "val_samples": val_size, "classes": list(classes)},
        "train": {
            "epochs": cfg.epochs,
            "start_epoch": start_epoch,
            "batch_size": cfg.batch_size,
            "img_size": cfg.img_size,
            "disable_mix_last": cfg.disable_mix_last,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
            "unfreeze": cfg.unfreeze,
            "unfreeze_epoch": cfg.unfreeze_epoch,
            "lr_head": cfg.lr_head,
            "lr_backbone": cfg.lr_backbone,
            "weight_decay": cfg.weight_decay,
            "label_smoothing": cfg.label_smoothing,
            "mixup_alpha": cfg.mixup_alpha,
            "cutmix_alpha": cfg.cutmix_alpha,
            "mix_prob": cfg.mix_prob,
            "disable_mix_last": cfg.disable_mix_last,
            "img_size": cfg.img_size,
            "scheduler": cfg.scheduler,
            "early_stop_patience": cfg.early_stop_patience,
            "topk": cfg.topk,
            "write_metrics": cfg.write_metrics,
            "metrics_out": str(cfg.metrics_out),
        },
    }
    LOG.debug("Training configuration:\n%s", json.dumps(detail, indent=2))


# ----------------------------
# MixUp / CutMix
# ----------------------------

def _rand_bbox(h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
    # Standard CutMix bbox calculation
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # uniform center
    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    return x1, y1, x2, y2


def _apply_mix(inputs: torch.Tensor,
               targets: torch.Tensor,
               *,
               mixup_alpha: float,
               cutmix_alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Returns: mixed_inputs, targets_a, targets_b, lam
    """
    batch_size = inputs.size(0)
    if batch_size < 2:
        return inputs, targets, targets, 1.0

    index = torch.randperm(batch_size, device=inputs.device)
    targets_a = targets
    targets_b = targets[index]

    # Choose which augmentation to apply
    use_cutmix = (cutmix_alpha > 0) and (mixup_alpha <= 0 or random.random() < 0.5)

    if use_cutmix:
        alpha = cutmix_alpha
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
        _, _, h, w = inputs.shape
        x1, y1, x2, y2 = _rand_bbox(h, w, lam)

        inputs = inputs.clone()
        inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual area
        box_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - box_area / float(h * w)
        return inputs, targets_a, targets_b, lam

    # MixUp
    alpha = mixup_alpha
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    mixed = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed, targets_a, targets_b, lam


# ----------------------------
# Metrics
# ----------------------------

@torch.no_grad()
def _evaluate_and_confusion(model: nn.Module,
                            loader: DataLoader,
                            *,
                            device: torch.device,
                            num_classes: int,
                            pin_memory: bool,
                            autocast_ctx,
                            topk: int) -> Dict:
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total = 0
    correct1 = 0
    correctk = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        with autocast_ctx():
            outputs = model(inputs)

        # Top-1
        preds = outputs.argmax(dim=1)
        correct1 += (preds == labels).sum().item()

        # Top-K
        if topk > 1:
            topk_preds = outputs.topk(k=min(topk, outputs.size(1)), dim=1).indices
            correctk += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

        # Confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[int(t), int(p)] += 1

    eps = 1e-12
    tp = cm.diag().to(torch.float64)
    fp = cm.sum(dim=0).to(torch.float64) - tp
    fn = cm.sum(dim=1).to(torch.float64) - tp

    precision = (tp / (tp + fp + eps)).cpu().tolist()
    recall = (tp / (tp + fn + eps)).cpu().tolist()
    f1 = (2 * tp / (2 * tp + fp + fn + eps)).cpu().tolist()

    macro_f1 = float(torch.tensor(f1, dtype=torch.float64).mean().item())

    metrics = {
        "top1_acc": 100.0 * correct1 / max(1, total),
        "topk_acc": 100.0 * correctk / max(1, total) if topk > 1 else None,
        "confusion_matrix": cm.cpu().tolist(),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "macro_f1": macro_f1,
    }
    return metrics


def _top_confusions(cm: List[List[int]], top_n: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    n = len(cm)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i][j])
            if c > 0:
                out.append((i, j, c))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:max(0, top_n)]


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    cfg = parse_args()
    _setup_logging(cfg.debug)
    _set_seed(cfg.seed)

    if cfg.torch_threads and cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)

    device = _select_device(cfg.device)

    # AMP: enable only for CUDA (MPS autocast exists but often provides limited benefit and can be finicky)
    use_amp = device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    pin_memory = device.type == "cuda"

    # Data
    train_tf, val_tf = _build_transforms(cfg.img_size)
    train_dir = cfg.data_dir / "train"
    val_dir = cfg.data_dir / "test"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Expected {train_dir} and {val_dir} to exist (ImageFolder structure).")

    train_dataset = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = torchvision.datasets.ImageFolder(str(val_dir), transform=val_tf)

    if train_dataset.classes != val_dataset.classes:
        raise SystemExit("Train/test class folders differ. Ensure identical folder names under train/ and test/.")

    classes = tuple(train_dataset.classes)
    num_classes = len(classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # Model
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Phase A default (head-only), but we may override based on resume/start epoch below.
    _set_trainable(model, mode="head")

    # Criterion
    if cfg.label_smoothing and cfg.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer/Scheduler (created after resume because param trainability can change)
    optimizer = _make_optimizer(model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)

    def make_scheduler():
        if cfg.scheduler == "none":
            return None
        if cfg.scheduler == "cosine":
            # Cosine over total epochs
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=1e-6)
        # plateau
        mode = "min" if cfg.best_metric == "loss" else "max"
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=2, factor=0.5, verbose=cfg.debug
        )

    scheduler = make_scheduler()

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    best_val_acc = -1.0
    best_val_acc_epoch = 0
    epochs_no_improve = 0

    # Resume
    if cfg.resume and cfg.resume.is_file():
        ckpt = torch.load(str(cfg.resume), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                LOG.debug("Scheduler state could not be loaded; continuing with fresh scheduler state.")
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_loss_epoch = int(ckpt.get("best_val_loss_epoch", best_val_loss_epoch))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        best_val_acc_epoch = int(ckpt.get("best_val_acc_epoch", best_val_acc_epoch))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

    # Set trainable state based on start_epoch (important on resume and for consistent behavior)
    if start_epoch >= cfg.unfreeze_epoch:
        _set_trainable(model, mode=cfg.unfreeze)
    else:
        _set_trainable(model, mode="head")

    # Rebuild optimizer param groups because trainability may have changed after resume logic
    optimizer = _make_optimizer(model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
    # Scheduler depends on optimizer instance
    scheduler = make_scheduler()

    # Ensure checkpoints dir exists
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Write classmap.json
    _write_classmap(train_dataset, Path("classmap.json"))

    _print_config(cfg, device, pin_memory, use_amp, len(train_dataset), len(val_dataset), classes, start_epoch)

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        if epoch == cfg.unfreeze_epoch and start_epoch < cfg.unfreeze_epoch:
            LOG.info("Unfreezing backbone (%s) at epoch %d", cfg.unfreeze, epoch)
            _set_trainable(model, mode=cfg.unfreeze)
            optimizer = _make_optimizer(model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
            scheduler = make_scheduler()

        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            # MixUp / CutMix (train only)
            mixed = False
            lam = 1.0
            y_a = labels
            y_b = labels
            mix_active = (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0)
            if cfg.disable_mix_last > 0 and epoch >= (cfg.epochs - cfg.disable_mix_last):
                mix_active = False

            if mix_active and (random.random() < cfg.mix_prob):
                inputs, y_a, y_b, lam = _apply_mix(
                    inputs, labels, mixup_alpha=cfg.mixup_alpha, cutmix_alpha=cfg.cutmix_alpha
                )
                mixed = True

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                outputs = model(inputs)
                if mixed:
                    loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * inputs.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                with autocast_ctx():
                    outputs = model(inputs)
                    if cfg.tta:
                        outputs_flip = model(torch.flip(inputs, dims=[3]))
                        outputs = (outputs + outputs_flip) / 2.0
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * correct / max(1, total)

        # Scheduler step
        if scheduler is not None:
            if cfg.scheduler == "plateau":
                scheduler.step(val_loss)
            elif cfg.scheduler == "cosine":
                scheduler.step()

        LOG.info(
            "Epoch %d/%d — Train Loss: %.4f, Val Loss: %.4f, Val Acc: %.2f%%",
            epoch + 1, cfg.epochs, train_loss, val_loss, val_accuracy
        )

        # Save epoch checkpoint
        ckpt_path = ckpt_dir / f"resnet50_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_val_loss": best_val_loss,
                "best_val_loss_epoch": best_val_loss_epoch,
                "best_val_acc": best_val_acc,
                "best_val_acc_epoch": best_val_acc_epoch,
                "epochs_no_improve": epochs_no_improve,
                "best_metric": cfg.best_metric,

                "classes": list(classes),
            },
            ckpt_path,
        )

        # Track best checkpoints (loss + accuracy) regardless of early-stop metric
        improved_loss = val_loss < (best_val_loss - 1e-6)
        if improved_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        improved_acc = val_accuracy > (best_val_acc + 1e-6)
        if improved_acc:
            best_val_acc = val_accuracy
            best_val_acc_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_dir / "best_acc_model.pth")

        # Early-stopping & scheduler monitoring metric
        monitor_value = val_loss if cfg.best_metric == "loss" else val_accuracy
        monitor_name = "val_loss" if cfg.best_metric == "loss" else "val_acc"
        monitor_improved = improved_loss if cfg.best_metric == "loss" else improved_acc

        if monitor_improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if cfg.early_stop_patience and cfg.early_stop_patience > 0:
            if epochs_no_improve >= cfg.early_stop_patience:
                LOG.info(
                    "Early stopping triggered (no %s improvement for %d epochs).",
                    monitor_name, cfg.early_stop_patience
                )
                break

    # Metrics export (best model)
    if cfg.write_metrics:
        best_metric = getattr(cfg, "best_metric", "loss")
        best_name = "best_model.pth" if best_metric == "loss" else "best_acc_model.pth"
        best_path = ckpt_dir / best_name
        if best_path.is_file():
            model.load_state_dict(torch.load(str(best_path), map_location="cpu"))
            model.to(device)
        metrics = _evaluate_and_confusion(
            model, val_loader, device=device, num_classes=num_classes, pin_memory=pin_memory,
            autocast_ctx=autocast_ctx, topk=cfg.topk
        )
        confusions = _top_confusions(metrics["confusion_matrix"], cfg.top_confusions)
        metrics["top_confusions"] = confusions
        metrics["classes"] = list(classes)
        metrics["best_val_loss"] = best_val_loss
        metrics["best_val_loss_epoch"] = best_val_loss_epoch
        metrics["best_val_acc"] = best_val_acc
        metrics["best_val_acc_epoch"] = best_val_acc_epoch
        metrics["best_metric"] = getattr(cfg, "best_metric", "loss")
        metrics["config"] = {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr_head": cfg.lr_head,
            "lr_backbone": cfg.lr_backbone,
            "weight_decay": cfg.weight_decay,
            "unfreeze": cfg.unfreeze,
            "unfreeze_epoch": cfg.unfreeze_epoch,
            "label_smoothing": cfg.label_smoothing,
            "mixup_alpha": cfg.mixup_alpha,
            "cutmix_alpha": cfg.cutmix_alpha,
            "mix_prob": cfg.mix_prob,
            "scheduler": cfg.scheduler,
            "early_stop_patience": cfg.early_stop_patience,
            "topk": cfg.topk,
        }

        if cfg.write_metrics:
            out_path = cfg.metrics_out

            # If a run name is provided and the user left the default filename, include the run name.
            if cfg.run_name and out_path.name == "metrics.json":
                out_path = out_path.with_name(f"metrics_{cfg.run_name}.json")

            # Never overwrite an existing metrics file unless explicitly requested.
            if out_path.exists() and not cfg.overwrite_metrics:
                suffix = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(metrics, indent=2))
            LOG.info("Wrote metrics to %s", out_path)

            # Convenience pointer to the most recent run (always overwritten).
            latest_path = out_path.with_name("metrics_latest.json")
            try:
                latest_path.write_text(json.dumps(metrics, indent=2))
            except Exception as e:
                LOG.warning("Failed to write %s: %s", latest_path, e)

        # Console summary of worst confusions
        if confusions:
            LOG.info("Top confusions (true -> pred : count):")
            for t, p, c in confusions:
                LOG.info("  %s -> %s : %d", classes[t], classes[p], c)

    LOG.info("Training complete.")
    return 0


if __name__ == "__main__":
    # Portable multiprocessing behavior:
    # - macOS/Windows: spawn
    # - Linux: default for speed
    if platform.system().lower() in ("darwin", "windows"):
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    raise SystemExit(main())
