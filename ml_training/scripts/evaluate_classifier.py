#!/usr/bin/env python3
"""
Held-out Test Evaluation Script for Multi-Crop Disease Classifiers (Phase 3A).
Evaluates an EfficientNet-B0 checkpoint ONLY on the held-out test split.

Features:
- Sugarcane: evaluates strictly on deduplicated evaluation rows (use_for_eval == true).
- Calculates accuracy, per-class precision, recall, F1, support, and macro-F1.
- Computes Wilson 95% confidence intervals for per-class recall.
- Generates confusion matrix heatmap and recall error-bar chart (PNG).
- Records all misclassified test files with ground truth, prediction, and confidence.
- Outputs structured report JSON and plots to reports/ directory.
- For cotton: explicitly tags test metrics as 'indicative only (tiny classes: curl virus test n=2, fusarium wilt test n=5)'.

Rules:
- No fake data, no invented numbers.
- Evaluates strictly on real model checkpoint predictions over physical test images.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ml_training.scripts.classifier_utils import (
    CROP_MANIFEST_MAP,
    CropDataset,
    compute_classification_metrics,
    get_default_data_root,
    load_manifest_rows,
    resolve_image_path,
)


def get_eval_transform(img_size: int = 224) -> transforms.Compose:
    """Build deterministic evaluation transform."""
    return transforms.Compose([
        transforms.Resize(int(round(img_size * 256 / 224))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def plot_evaluation_charts(
    metrics: Dict,
    class_names: List[str],
    crop_name: str,
    output_png: Path,
    is_smoke: bool = False,
) -> None:
    """
    Generate two-panel evaluation figure:
    Panel 1: Confusion matrix heatmap.
    Panel 2: Per-class recall with Wilson 95% confidence interval error bars.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    title_suffix = " [SMOKE TEST, not valid metrics]" if is_smoke else ""
    if is_smoke:
        fig.suptitle("SMOKE TEST, not valid metrics", fontsize=13, color="red", fontweight="bold")

    cm = np.array(metrics["confusion_matrix"])

    # Panel 1: Confusion Matrix
    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=ax1
        )
    else:
        im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax1)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax1.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(class_names)))
        ax1.set_xticklabels(class_names)
        ax1.set_yticklabels(class_names)

    ax1.set_title(f"{crop_name.capitalize()} - Test Confusion Matrix{title_suffix}")
    ax1.set_ylabel("True Disease Class")
    ax1.set_xlabel("Predicted Disease Class")
    ax1.tick_params(axis="x", rotation=30)

    # Panel 2: Per-class Recall with Wilson 95% CIs
    recalls = [metrics["per_class"][c]["recall"] for c in class_names]
    lower_errors = []
    upper_errors = []
    for c in class_names:
        r = metrics["per_class"][c]["recall"]
        ci = metrics["per_class"][c]["recall_wilson_95_ci"]
        lower_errors.append(max(0.0, r - ci[0]))
        upper_errors.append(max(0.0, ci[1] - r))

    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, recalls, xerr=[lower_errors, upper_errors], align="center",
             color="#2b5c8f", ecolor="#e74c3c", capsize=5, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlim(0.0, 1.05)
    ax2.set_xlabel("Recall (with Wilson 95% CI Error Bars)")
    ax2.set_title(f"{crop_name.capitalize()} - Per-Class Recall & 95% CIs{title_suffix}")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Annotate values
    for i, (r, le, ue) in enumerate(zip(recalls, lower_errors, upper_errors)):
        sup = metrics["per_class"][class_names[i]]["support"]
        ax2.text(max(0.05, r / 2), i, f"{r*100:.1f}% (n={sup})",
                 va="center", ha="center", color="white", fontweight="bold")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    crop: Optional[str] = None,
    out_dir: Optional[Path] = None,
    batch_size: int = 32,
    device_str: str = "auto",
    smoke: bool = False,
) -> Dict:
    """
    Load an EfficientNet-B0 checkpoint, run inference on the test split,
    and save full evaluation reports and figures.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Determine device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Smoke test detection
    is_smoke = smoke or bool(ckpt.get("is_smoke_test") or ckpt.get("config", {}).get("is_smoke_test"))
    max_eval_batches = 3 if is_smoke else None
    if is_smoke:
        print("\n" + "!" * 80)
        print(" [SMOKE TEST EVALUATION]: Processing at most 3 batches.")
        print(" Report and charts will be labelled: 'SMOKE TEST, not valid metrics'.")
        print("!" * 80)

    # Detect crop
    crop_name = (crop or ckpt.get("crop") or "unknown").lower()
    config = ckpt.get("config", {})
    img_size = config.get("img_size", 224)

    # Detect classes
    class_names = ckpt.get("classes")
    class_to_idx = ckpt.get("class_to_idx")
    if not class_names:
        if class_to_idx:
            class_names = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
        else:
            raise ValueError("No class definitions found in checkpoint.")
    num_classes = len(class_names)

    # Resolve manifest
    if manifest_path is None:
        if "manifest" in config and Path(config["manifest"]).exists():
            manifest_path = Path(config["manifest"])
        else:
            m_name = CROP_MANIFEST_MAP.get(crop_name)
            manifest_path = Path("ml_training/manifests") / m_name
    print(f"[INFO] Using manifest: {manifest_path}")

    # Resolve data root
    if data_root is None:
        if "data_root" in config and Path(config["data_root"]).exists():
            data_root = Path(config["data_root"])
        else:
            data_root = get_default_data_root()
    print(f"[INFO] Using data root: {data_root}")

    # Load test split rows
    test_rows = load_manifest_rows(
        manifest_path=manifest_path,
        crop=crop_name,
        split="test",
        filter_eval_only=True,
    )
    print(f"[INFO] Filtered test samples: {len(test_rows)} images (Deduplication filter active for sugarcane)")

    # Resolve physical samples: (abs_path, label_idx, rel_path)
    samples = []
    for r in test_rows:
        p = resolve_image_path(data_root, crop_name, r["path"])
        if not p.exists():
            raise FileNotFoundError(f"Image not found on disk: {p} (resolved from {r['path']})")
        samples.append((p, class_to_idx[r["class"]], r["path"]))

    # Prepare Dataset & DataLoader
    eval_tf = get_eval_transform(img_size)
    dataset_samples = [(p, idx) for p, idx, _ in samples]
    test_ds = CropDataset(dataset_samples, transform=eval_tf)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0 if sys.platform.startswith("win") else 2,
        pin_memory=(device.type == "cuda")
    )

    # Build model & load state dict
    if not HAS_TIMM:
        raise ImportError("timm package is required. Install with: pip install timm")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Inference loop
    all_preds = []
    all_targets = []
    all_probs = []

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(all_targets, all_preds, class_names)

    # Identify misclassified test files
    misclassified = []
    evaluated_samples = samples[:len(all_preds)]
    for idx, (p_abs, t_idx, rel_p) in enumerate(evaluated_samples):
        pred_idx = all_preds[idx]
        if pred_idx != t_idx:
            prob_dict = {class_names[c_i]: round(all_probs[idx][c_i], 4) for c_i in range(num_classes)}
            misclassified.append({
                "path": rel_p,
                "true_class": class_names[t_idx],
                "pred_class": class_names[pred_idx],
                "confidence": round(all_probs[idx][pred_idx], 4),
                "probabilities": prob_dict,
            })

    # Prepare output filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir or "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = out_dir / f"metrics_{crop_name}_{timestamp}.json"
    report_png_path = out_dir / f"evaluation_{crop_name}_{timestamp}.png"

    # Assemble report payload
    full_report = {
        "crop": crop_name,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "timestamp": timestamp,
        "evaluation_split": "test",
        "total_test_samples": len(evaluated_samples),
        "total_misclassified": len(misclassified),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": class_names,
        "misclassified_files": misclassified,
        "indicative_only_warning": (
            "tiny classes: curl virus test n=2, fusarium wilt test n=5" if crop_name == "cotton" else None
        ),
        "is_smoke_test": is_smoke,
        "smoke_test": "SMOKE TEST, not valid metrics" if is_smoke else None,
    }

    # Save JSON report
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"[SUCCESS] Saved evaluation JSON metrics to: {report_json_path}")

    # Save PNG figure
    plot_evaluation_charts(metrics, class_names, crop_name, report_png_path, is_smoke=is_smoke)
    print(f"[SUCCESS] Saved evaluation charts to: {report_png_path}")

    # Real Console Output Table (No numbers typed by hand)
    print("\n" + "=" * 90)
    print(f" EVALUATION REPORT: {crop_name.upper()} DISEASE CLASSIFIER (EfficientNet-B0)")
    if is_smoke:
        print(" *** [WARNING] THIS RUN WAS A SMOKE TEST: ALL NUMBERS ARE 'SMOKE TEST, not valid metrics' ***")
    if crop_name == "cotton":
        print(" [NOTE] Test result labelled: indicative only (tiny classes: curl virus test n=2, fusarium wilt test n=5)")
    print("=" * 90)
    print(f"Test Samples: {len(samples)}  |  Accuracy: {metrics['accuracy']*100:.2f}%  |  Macro-F1: {metrics['macro_f1']*100:.2f}%  |  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"Misclassified: {len(misclassified)} / {len(samples)} ({len(misclassified)/len(samples)*100:.2f}%)")
    print("-" * 90)
    print(f"{'Disease Class':<22} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7} | {'Wilson 95% Recall CI':>22}")
    print("-" * 90)
    for c in class_names:
        cd = metrics["per_class"][c]
        ci = cd["recall_wilson_95_ci"]
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"{c:<22} | {cd['precision']:>9.4f} | {cd['recall']:>9.4f} | {cd['f1']:>9.4f} | {cd['support']:>7d} | {ci_str:>22}")
    print("-" * 90)

    # Confusion matrix printed
    print("\n[CONFUSION MATRIX (Rows=True, Columns=Predicted)]")
    hdr = f"{'True \\ Pred':<22} | " + " | ".join(f"{c[:10]:>10}" for c in class_names)
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(class_names):
        row_str = " | ".join(f"{metrics['confusion_matrix'][i][j]:>10d}" for j in range(num_classes))
        print(f"{c:<22} | {row_str}")
    print("-" * len(hdr))

    if misclassified:
        print(f"\n[MISCLASSIFIED TEST SAMPLES (N={len(misclassified)})]:")
        for m in misclassified[:10]:
            print(f"  * {m['path']} -> True: '{m['true_class']}', Predicted: '{m['pred_class']}' (Conf: {m['confidence']:.4f})")
        if len(misclassified) > 10:
            print(f"  ... and {len(misclassified) - 10} more (see JSON report)")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Disease Classifier on Held-out Test Split.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to dataset manifest CSV (optional, inferred from checkpoint).")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Directory containing raw image datasets.")
    parser.add_argument("--crop", type=str, default=None, choices=["sugarcane", "wheat", "cotton"],
                        help="Crop name (optional, inferred from checkpoint).")
    parser.add_argument("--out-dir", type=str, default="reports",
                        help="Directory to save metric JSON and PNG reports (default: reports).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for test evaluation (default: 32).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use for evaluation (default: auto).")
    parser.add_argument("--smoke", action="store_true",
                        help="Run quick smoke test evaluation (at most 3 batches, labelled: SMOKE TEST, not valid metrics).")

    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest) if args.manifest else None,
        data_root=Path(args.data_root) if args.data_root else None,
        crop=args.crop,
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
        device_str=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
