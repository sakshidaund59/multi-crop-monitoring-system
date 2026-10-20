#!/usr/bin/env python3
"""
EfficientNet-B0 Disease Classifier Training Script for Multi-Crop Monitoring System (Phase 3A).
Supports Sugarcane (SPPU), Wheat (Zenodo/JIC), and Cotton (Mendeley).

Features:
- PyTorch + timm efficientnet_b0 with ImageNet pretrained weights.
- On-the-fly data augmentation on train split only (never writes augmented images to disk).
- Sugarcane: evaluates val/test strictly on deduplicated rows (use_for_eval == true).
- Cotton: uses train-split class weights in CrossEntropyLoss; performs stratified 5-fold CV on
  train+val (882 images) and reports per-class recall mean +/- std; reports test split labelled
  'indicative only (tiny classes: curl virus test n=2, fusarium wilt test n=5)'.
- Fixed random seeds for 100% reproducibility.
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Early stopping on validation macro-F1.
- Mixed precision training on GPU (torch.cuda.amp / torch.amp).
- Epoch-by-epoch checkpointing (best_checkpoint.pt and last_checkpoint.pt) with disconnect resume.
- Saves training curves (PNG + JSON), exact config.json, and classes.json.

Rules:
- No fake data, no invented metrics.
- Designed for Google Colab T4 GPU execution.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# Import shared classifier utilities
from ml_training.scripts.classifier_utils import (
    CROP_MANIFEST_MAP,
    CROP_METADATA_MAP,
    CropDataset,
    compute_classification_metrics,
    detect_smoke_artifacts,
    get_class_weights,
    get_default_data_root,
    load_manifest_rows,
    resolve_image_path,
)


def set_seed(seed: int = 42) -> None:
    """Set seeds across random, numpy, and torch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train augmentation transform and validation/test deterministic transform.
    Augmentation on TRAIN only: flip, rotation, colour jitter, random resized crop.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(round(img_size * 256 / 224))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate timm efficientnet_b0 with ImageNet pretrained weights."""
    if not HAS_TIMM:
        raise ImportError("timm package is required for training. Install with: pip install timm")
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Run one training epoch with mixed precision support."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda_amp = (device.type == "cuda" and scaler is not None)

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_cuda_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: List[str],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict]:
    """Evaluate model on validation or test DataLoader."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    eval_loss = running_loss / max(1, total)
    metrics = compute_classification_metrics(all_targets, all_preds, class_names)
    return eval_loss, metrics


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    crop_name: str,
    is_smoke: bool = False,
) -> None:
    """Generate and save PNG curves for loss and macro-F1."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    title_suffix = " [SMOKE TEST, not valid metrics]" if is_smoke else ""
    if is_smoke:
        fig.suptitle("SMOKE TEST, not valid metrics", fontsize=13, color="red", fontweight="bold")

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", markersize=3)
    ax1.set_title(f"{crop_name.capitalize()} - Loss Curves{title_suffix}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy & Macro-F1 subplot
    ax2.plot(epochs, history["train_acc"], "g-o", label="Train Accuracy", markersize=3)
    ax2.plot(epochs, history["val_acc"], "m--s", label="Val Accuracy", markersize=3)
    ax2.plot(epochs, history["val_macro_f1"], "k-^", label="Val Macro-F1", markersize=3)
    ax2.set_title(f"{crop_name.capitalize()} - Accuracy & Macro-F1{title_suffix}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_stratified_5fold_cv_cotton(
    train_val_samples: List[Tuple[Path, int]],
    class_names: List[str],
    class_weights: Optional[torch.Tensor],
    args,
    device: torch.device,
) -> Dict[str, Tuple[float, float]]:
    """
    Perform 5-fold stratified cross-validation on train+val (882 images) for Cotton.
    Reports per-class recall mean +/- std across folds.
    """
    print("\n" + "=" * 75)
    print(" COTTON: RUNNING STRATIFIED 5-FOLD CROSS-VALIDATION ON TRAIN+VAL (N=882)")
    print("=" * 75)

    num_classes = len(class_names)
    # Stratify by class label
    class_buckets = defaultdict(list)
    for idx, (p, c) in enumerate(train_val_samples):
        class_buckets[c].append(idx)

    # Deterministic split into 5 folds
    rng = random.Random(args.seed)
    fold_val_indices = [[] for _ in range(5)]
    for c, idxs in sorted(class_buckets.items()):
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        for i, idx in enumerate(shuffled):
            fold_val_indices[i % 5].append(idx)

    train_tf, eval_tf = get_transforms(args.img_size)
    fold_per_class_recall = defaultdict(list)

    for fold in range(5):
        val_idx_set = set(fold_val_indices[fold])
        tr_idx_list = [i for i in range(len(train_val_samples)) if i not in val_idx_set]
        va_idx_list = list(val_idx_set)

        tr_subset = [train_val_samples[i] for i in tr_idx_list]
        va_subset = [train_val_samples[i] for i in va_idx_list]

        tr_ds = CropDataset(tr_subset, transform=train_tf)
        va_ds = CropDataset(va_subset, transform=eval_tf)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = build_model(num_classes=num_classes, pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, args.epochs // 2), eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_f1 = -1.0
        best_fold_recall = {}

        max_cv_batches = 3 if getattr(args, "smoke", False) else None
        cv_epochs = 1 if getattr(args, "smoke", False) else min(15, args.epochs)
        for ep in range(1, cv_epochs + 1):
            train_one_epoch(model, tr_loader, criterion, optimizer, scaler, device, max_batches=max_cv_batches)
            scheduler.step()
            _, val_metrics = evaluate_epoch(model, va_loader, criterion, class_names, device, max_batches=max_cv_batches)
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                best_fold_recall = {c: val_metrics["per_class"][c]["recall"] for c in class_names}

        for c in class_names:
            fold_per_class_recall[c].append(best_fold_recall.get(c, 0.0))

        print(f"  Fold {fold + 1}/5: Val Macro-F1 = {best_f1:.4f}")

    # Compute mean +/- std
    cv_summary = {}
    print("\n[COTTON 5-FOLD STRATIFIED CV RESULTS (TRAIN+VAL)]")
    print(f"{'Disease Class':<22} | {'Recall (Mean +/- Std)':<25} | {'Fold 1..5 Recalls':<30}")
    print("-" * 82)
    for c in class_names:
        rec_list = fold_per_class_recall[c]
        mean_r = float(np.mean(rec_list))
        std_r = float(np.std(rec_list))
        cv_summary[c] = (round(mean_r, 4), round(std_r, 4))
        rec_str = ", ".join(f"{r:.3f}" for r in rec_list)
        print(f"{c:<22} | {mean_r * 100:>5.1f}% +/- {std_r * 100:>4.1f}%          | [{rec_str}]")
    print("-" * 82)
    return cv_summary


def train_classifier(args):
    """Main training workflow with Colab disconnection recovery."""
    set_seed(args.seed)

    crop = args.crop.lower()
    data_root = Path(args.data_root) if args.data_root else get_default_data_root()
    out_dir = Path(args.out_dir)

    # Smoke test mode configuration and directory isolation
    is_smoke = bool(getattr(args, "smoke", False))
    if is_smoke:
        # A smoke run must write to its own folder (e.g. <out-dir>/_smoke_<crop>)
        # and must never write into or read from the real run folder.
        if out_dir.name != f"_smoke_{crop}" and not out_dir.name.startswith(f"_smoke_{crop}"):
            out_dir = out_dir / f"_smoke_{crop}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "!" * 75)
        print(" [SMOKE TEST MODE ENABLED]: 1 epoch, at most 3 batches per epoch.")
        print(f" Dedicated Smoke Run Directory: {out_dir}")
        print(" All output files will be labelled: 'SMOKE TEST, not valid metrics'.")
        print("!" * 75)
        args.epochs = 1
    else:
        # Real training run: MUST refuse to start in a folder containing smoke artifacts
        smoke_reasons = detect_smoke_artifacts(out_dir)
        if smoke_reasons:
            raise ValueError(
                f"Cannot start real training run in '{out_dir}': folder contains smoke artifacts ({'; '.join(smoke_reasons)}). "
                "Real training runs must not be executed in folders containing smoke artifacts."
            )
        out_dir.mkdir(parents=True, exist_ok=True)

    max_batches = 3 if is_smoke else None

    # Check resume safety early: refuse if checkpoint is from a smoke run
    last_ckpt_path = out_dir / "last_checkpoint.pt"
    if getattr(args, "resume", False) and last_ckpt_path.exists():
        ckpt_meta = torch.load(last_ckpt_path, map_location="cpu")
        if ckpt_meta.get("is_smoke_test") or ckpt_meta.get("config", {}).get("is_smoke_test"):
            raise ValueError(
                f"Cannot resume training: checkpoint at '{last_ckpt_path}' is from a smoke test run "
                "('is_smoke_test' is true). Refusing to resume from smoke artifacts."
            )

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Locate manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        m_name = CROP_MANIFEST_MAP.get(crop)
        manifest_path = Path("ml_training/manifests") / m_name
    print(f"[INFO] Loading manifest: {manifest_path}")

    # Locate metadata
    meta_name = CROP_METADATA_MAP.get(crop)
    meta_path = Path("ml_training/manifests") / meta_name

    # Load splits
    # Train split: no eval filter
    train_rows = load_manifest_rows(manifest_path, crop=crop, split="train", filter_eval_only=False)
    # Val split: deduplicated rows only for sugarcane
    val_rows = load_manifest_rows(manifest_path, crop=crop, split="val", filter_eval_only=True)
    # Test split: deduplicated rows only for sugarcane
    test_rows = load_manifest_rows(manifest_path, crop=crop, split="test", filter_eval_only=True)

    print(f"[INFO] Loaded split counts: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")

    # Extract distinct classes deterministically
    all_classes = sorted(list(set(r["class"] for r in train_rows + val_rows + test_rows)))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    num_classes = len(all_classes)
    print(f"[INFO] Target classes ({num_classes}): {all_classes}")

    # Save classes.json
    classes_payload = {"classes": all_classes, "class_to_idx": class_to_idx}
    if is_smoke:
        classes_payload["smoke_test"] = "SMOKE TEST, not valid metrics"
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes_payload, f, indent=2)

    # Resolve samples: (abs_path, label_idx)
    def to_samples(rows: List[Dict]) -> List[Tuple[Path, int]]:
        samples = []
        for r in rows:
            p = resolve_image_path(data_root, crop, r["path"])
            if not p.exists():
                raise FileNotFoundError(f"Image not found on disk: {p} (resolved from {r['path']})")
            samples.append((p, class_to_idx[r["class"]]))
        return samples

    train_samples = to_samples(train_rows)
    val_samples = to_samples(val_rows)
    test_samples = to_samples(test_rows)

    # Class weights for loss
    class_weights = None
    if crop == "cotton":
        class_weights = get_class_weights(meta_path, class_to_idx)
        if class_weights is not None:
            print(f"[INFO] Loaded Cotton train class weights: {class_weights.tolist()}")

    # Optional 5-Fold Cross Validation for Cotton on train+val
    cv_summary = None
    if crop == "cotton" or getattr(args, "cv", False):
        cv_samples = train_samples + val_samples
        cv_summary = run_stratified_5fold_cv_cotton(
            cv_samples, all_classes, class_weights, args, device
        )

    # Data transforms & DataLoaders
    train_tf, eval_tf = get_transforms(args.img_size)
    train_ds = CropDataset(train_samples, transform=train_tf)
    val_ds = CropDataset(val_samples, transform=eval_tf)
    test_ds = CropDataset(test_samples, transform=eval_tf)

    num_workers = getattr(args, "num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    # Build Model
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # Loss, Optimizer, Scheduler, Scaler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 1e-2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Trackers
    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    # Resumption handling
    last_ckpt_path = out_dir / "last_checkpoint.pt"
    if args.resume:
        if last_ckpt_path.exists():
            print(f"[INFO] Checking checkpoint for resume: {last_ckpt_path}")
            ckpt = torch.load(last_ckpt_path, map_location=device)
            # Refuse to resume if checkpoint is from a smoke test run
            if ckpt.get("is_smoke_test") or ckpt.get("config", {}).get("is_smoke_test"):
                raise ValueError(
                    f"Cannot resume training: checkpoint at '{last_ckpt_path}' is from a smoke test run "
                    "('is_smoke_test' is true). Refusing to resume from smoke artifacts."
                )
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
            best_epoch = ckpt.get("best_epoch", -1)
            history = ckpt.get("history", history)
            no_improve_epochs = ckpt.get("no_improve_epochs", 0)
            print(f"[INFO] Resumed at epoch {start_epoch} (Best Val Macro-F1 so far: {best_val_macro_f1:.4f} at epoch {best_epoch})")
        else:
            print(f"[INFO] Resume requested, but no checkpoint found at '{last_ckpt_path}'. Starting from scratch.")

    # Save initial config.json
    config_dict = {
        "crop": crop,
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "img_size": args.img_size,
        "num_classes": num_classes,
        "classes": all_classes,
        "class_to_idx": class_to_idx,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "device": str(device),
        "cv_summary": cv_summary,
        "is_smoke_test": is_smoke,
        "smoke_test": "SMOKE TEST, not valid metrics" if is_smoke else None,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("\n" + "=" * 75)
    print(f" STARTING TRAINING: {crop.upper()} ({args.epochs} EPOCHS)")
    print("=" * 75)

    best_ckpt_path = out_dir / "best_checkpoint.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, max_batches=max_batches
        )
        scheduler.step()

        val_loss, val_metrics = evaluate_epoch(
            model, val_loader, criterion, all_classes, device, max_batches=max_batches
        )
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        elapsed = time.time() - t0

        # Record history
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))
        history["lr"].append(round(cur_lr, 6))

        # Check for improvement
        improved = (val_macro_f1 > best_val_macro_f1)
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_epochs = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_val_macro_f1": best_val_macro_f1,
                "best_epoch": best_epoch,
                "history": history,
                "config": config_dict,
                "class_to_idx": class_to_idx,
                "classes": all_classes,
                "crop": crop,
                "is_smoke_test": is_smoke,
                "smoke_test": "SMOKE TEST, not valid metrics" if is_smoke else None,
            }, best_ckpt_path)
        else:
            no_improve_epochs += 1

        # Always save last checkpoint for disconnection resume
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch,
            "history": history,
            "config": config_dict,
            "class_to_idx": class_to_idx,
            "classes": all_classes,
            "crop": crop,
            "no_improve_epochs": no_improve_epochs,
            "is_smoke_test": is_smoke,
            "smoke_test": "SMOKE TEST, not valid metrics" if is_smoke else None,
        }, last_ckpt_path)

        # Save training curves incrementally
        if is_smoke:
            history["smoke_test"] = "SMOKE TEST, not valid metrics"
        with open(out_dir / "training_curves.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, out_dir / "training_curves.png", crop, is_smoke=is_smoke)

        tag = "(*BEST*)" if improved else ""
        print(f"Epoch {epoch:2d}/{args.epochs:2d} [{elapsed:4.1f}s] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}% Macro-F1: {val_macro_f1*100:5.1f}% {tag}")

        # Early stopping
        if no_improve_epochs >= getattr(args, "patience", 10):
            print(f"\n[INFO] Early stopping triggered after {no_improve_epochs} epochs without validation macro-F1 improvement.")
            break

    print(f"\n[TRAINING FINISHED] Best Validation Macro-F1: {best_val_macro_f1*100:.2f}% at epoch {best_epoch}")
    print(f"[INFO] Best model weights saved to: {best_ckpt_path}")

    # Held-out Test Split Final Evaluation
    print("\n" + "=" * 75)
    test_header = f" HELD-OUT TEST EVALUATION ({crop.upper()})"
    if crop == "cotton":
        test_header += " — Indicative only (tiny classes: curl virus test n=2, fusarium wilt test n=5)"
    print(test_header)
    print("=" * 75)

    # Load best weights
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics = evaluate_epoch(model, test_loader, criterion, all_classes, device, max_batches=max_batches)
    if is_smoke:
        test_metrics["is_smoke_test"] = True
        test_metrics["smoke_test"] = "SMOKE TEST, not valid metrics"
        print("\n" + "!" * 88)
        print(" [WARNING] THIS RUN WAS A SMOKE TEST (1 epoch, max 3 batches).")
        print(" ALL OUTPUT METRICS ARE LABELED: 'SMOKE TEST, not valid metrics'.")
        print("!" * 88)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Test Macro-F1: {test_metrics['macro_f1']*100:.2f}%")
    print(f"{'Class Name':<22} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7} | {'Wilson 95% Recall CI':>22}")
    print("-" * 88)
    for c in all_classes:
        cm_data = test_metrics["per_class"][c]
        ci = cm_data["recall_wilson_95_ci"]
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"{c:<22} | {cm_data['precision']:>9.4f} | {cm_data['recall']:>9.4f} | {cm_data['f1']:>9.4f} | {cm_data['support']:>7d} | {ci_str:>22}")
    print("-" * 88)

    # Return summary dict
    return {
        "crop": crop,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Disease Classifier on Crop Datasets.")
    parser.add_argument("--crop", type=str, required=True, choices=["sugarcane", "wheat", "cotton"],
                        help="Target crop disease classifier to train.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest CSV. Defaults to ml_training/manifests/<crop_manifest>.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Directory containing raw image datasets.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save checkpoints and training curves. Defaults to checkpoints/<crop>.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training and evaluation (default: 32).")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate for AdamW (default: 5e-4).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input square image resolution (default: 224).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use for training (default: auto).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_checkpoint.pt if found.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience epochs on validation macro-F1 (default: 10).")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay for AdamW (default: 1e-2).")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform.startswith("win") else 2,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--cv", action="store_true",
                        help="Run 5-fold stratified cross validation on train+val.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 1-epoch smoke test with at most 3 batches per epoch (labelled: SMOKE TEST, not valid metrics).")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f"checkpoints/{args.crop.lower()}"

    train_classifier(args)


if __name__ == "__main__":
    main()
