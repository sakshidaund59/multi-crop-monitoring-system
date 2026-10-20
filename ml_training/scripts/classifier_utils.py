#!/usr/bin/env python3
"""
Classifier Utilities for Multi-Crop Monitoring System (Phase 3A).
Handles dataset path resolution, manifest loading, class weight extraction,
Wilson 95% score confidence intervals, and evaluation metrics computation.

Rules:
- No fake data, no invented numbers.
- Path-agnostic: normalises backslashes to forward slashes across Windows, Linux, and Colab.
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

# Default crop to dataset directory mapping under DATA_ROOT
CROP_DATASET_MAP = {
    "sugarcane": "sugarcane_sppu",
    "wheat": "wheat_zenodo_jic",
    "cotton": "cotton_mendeley",
    "rice": "rice_pest_lakthuru",
}

# Default manifest files under ml_training/manifests
CROP_MANIFEST_MAP = {
    "sugarcane": "sugarcane_sppu_manifest.csv",
    "wheat": "wheat_zenodo_jic_manifest.csv",
    "cotton": "cotton_mendeley_manifest.csv",
    "rice": "rice_pest_lakthuru_groupsplit_v2_manifest.csv",
}

CROP_METADATA_MAP = {
    "sugarcane": "sugarcane_sppu_metadata.json",
    "wheat": "wheat_zenodo_jic_metadata.json",
    "cotton": "cotton_mendeley_metadata.json",
    "rice": "rice_pest_lakthuru_groupsplit_v2_metadata.json",
}


def get_default_data_root() -> Path:
    """Return data root directory from DATA_ROOT env or default ml_training/datasets/raw."""
    env_root = os.environ.get("DATA_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    candidates = [
        Path("ml_training/datasets/raw"),
        Path("E:/datasets_raw"),
        Path("/content/data"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("ml_training/datasets/raw")


def resolve_image_path(data_root: Path, crop_or_dataset: str, raw_path: str) -> Path:
    """
    Resolve an image file path relative to data_root across platforms.
    Normalises backslashes to forward slashes.

    Search priority:
    1. data_root / dataset_name / rel_path (e.g. data_root / 'sugarcane_sppu' / 'Healthy/healthy (1).jpeg')
    2. data_root / crop / rel_path (e.g. data_root / 'sugarcane' / 'Healthy/healthy (1).jpeg')
    3. data_root / rel_path (e.g. when data_root is pointed directly at the dataset folder)
    """
    rel = Path(raw_path.replace("\\", "/"))
    ds_name = CROP_DATASET_MAP.get(crop_or_dataset.lower(), crop_or_dataset)

    # 1. Under dataset folder name
    p1 = data_root / ds_name / rel
    if p1.exists():
        return p1

    # 2. Under crop name
    p2 = data_root / crop_or_dataset.lower() / rel
    if p2.exists():
        return p2

    # 3. Direct relative path
    p3 = data_root / rel
    if p3.exists():
        return p3

    # Fallback to p1 for informative FileNotFoundError
    return p1


def load_manifest_rows(
    manifest_path: Path,
    crop: str,
    split: Optional[str] = None,
    filter_eval_only: bool = False,
) -> List[Dict]:
    """
    Load rows from a manifest CSV with optional split and use_for_eval filtering.
    For sugarcane: if filter_eval_only=True and split in ('val', 'test'), only
    rows where use_for_eval is true ('true', '1', 'yes') are retained.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        # Ignore comments
        lines = [line for line in f if not line.startswith("#")]
        reader = csv.DictReader(lines)
        for r in reader:
            row_split = r.get("split", "").strip().lower()
            if split is not None and row_split != split.lower():
                continue

            # Sugarcane evaluation deduplication filter
            if crop.lower() == "sugarcane" and filter_eval_only and row_split in ("val", "test"):
                eval_flag = str(r.get("use_for_eval", "true")).strip().lower()
                if eval_flag not in ("true", "1", "yes"):
                    continue

            # Normalise path
            r["path"] = r["path"].replace("\\", "/")
            rows.append(r)

    return rows


def get_class_weights(
    metadata_path: Path,
    class_to_idx: Dict[str, int],
) -> Optional[torch.Tensor]:
    """
    Load train-split class weights from dataset metadata JSON.
    Aligns the weights vector to the class_to_idx indexing order.
    """
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights_dict = data.get("train_class_weights")
        if not weights_dict:
            return None

        num_classes = len(class_to_idx)
        weights = [1.0] * num_classes
        for c_name, w in weights_dict.items():
            if c_name in class_to_idx:
                weights[class_to_idx[c_name]] = float(w)

        return torch.tensor(weights, dtype=torch.float32)
    except Exception as e:
        print(f"[WARNING] Could not load class weights from {metadata_path}: {e}")
        return None


def compute_wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion k / n.
    Standard z = 1.95996 for 95% confidence.

    Formula:
      center = (k + z^2 / 2) / (n + z^2)
      spread = (z / (n + z^2)) * sqrt(k * (n - k) / n + z^2 / 4)
      interval = [max(0.0, center - spread), min(1.0, center + spread)]

    Handles edge cases gracefully:
      - n == 0 -> (0.0, 0.0)
      - k == 0 -> (0.0, upper)
      - k == n -> (lower, 1.0)
    """
    if n <= 0:
        return (0.0, 0.0)

    # Standard two-tailed z-value lookup
    z_map = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_map.get(confidence, 1.959963984540054)
    z2 = z * z

    k = max(0, min(k, n))

    denom = n + z2
    center = (k + 0.5 * z2) / denom
    spread = (z / denom) * math.sqrt((k * (n - k) / n) + (0.25 * z2))

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (round(lower, 4), round(upper, 4))


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict:
    """
    Compute comprehensive classification metrics without external heavy dependencies:
    - Overall accuracy
    - Per-class precision, recall, f1, support
    - Wilson 95% confidence interval for per-class recall
    - Macro-averaged F1 and weighted-averaged F1
    - Confusion matrix (K x K list of lists)
    """
    num_classes = len(class_names)
    total_samples = len(y_true)

    if total_samples == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "total_samples": 0,
            "per_class": {},
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        }

    # Confusion Matrix: rows = true, cols = pred
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    # Overall Accuracy
    correct = sum(cm[i][i] for i in range(num_classes))
    accuracy = correct / total_samples

    per_class = {}
    f1_list = []
    support_list = []

    for idx, c_name in enumerate(class_names):
        tp = cm[idx][idx]
        fn = sum(cm[idx][j] for j in range(num_classes) if j != idx)
        fp = sum(cm[i][idx] for i in range(num_classes) if i != idx)
        support = tp + fn

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / support) if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Wilson 95% CI on recall (tp out of support)
        ci_lower, ci_upper = compute_wilson_score_interval(tp, support, confidence=0.95)

        per_class[c_name] = {
            "class_idx": idx,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "recall_wilson_95_ci": [ci_lower, ci_upper],
        }
        f1_list.append(f1)
        support_list.append(support)

    macro_f1 = sum(f1_list) / num_classes if num_classes > 0 else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_samples": total_samples,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


class CropDataset(Dataset):
    """
    PyTorch Dataset for crop disease images.
    Loads images on the fly via PIL and applies on-the-fly transforms.
    """
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            tensor = img_rgb
        return tensor, label


def detect_smoke_artifacts(folder: Path) -> List[str]:
    """
    Detect whether a directory contains smoke test artifacts or is a smoke test folder.
    Returns a list of reasons found, or an empty list if clean.
    """
    reasons = []
    if not folder.exists():
        return reasons

    # 1. Directory name check
    folder_name_lower = folder.name.lower()
    if folder_name_lower.startswith("_smoke") or folder_name_lower.startswith("smoke") or "_smoke_" in folder_name_lower:
        reasons.append(f"Directory name '{folder.name}' indicates a smoke test folder")

    # 2. Check for smoke subdirectories or smoke files
    try:
        for item in folder.iterdir():
            if item.is_dir() and (item.name.startswith("_smoke") or "_smoke_" in item.name.lower()):
                reasons.append(f"Contains smoke subdirectory '{item.name}'")
            elif item.is_file() and "_smoke" in item.name.lower():
                reasons.append(f"Contains smoke file '{item.name}'")
    except Exception:
        pass

    # 3. Check config.json
    cfg_path = folder / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if cfg.get("is_smoke_test") or cfg.get("smoke_test"):
                    reasons.append("config.json contains smoke test markings")
        except Exception:
            pass

    # 4. Check classes.json
    cls_path = folder / "classes.json"
    if cls_path.exists():
        try:
            with open(cls_path, "r", encoding="utf-8") as f:
                cls_data = json.load(f)
                if cls_data.get("smoke_test"):
                    reasons.append("classes.json contains smoke test markings")
        except Exception:
            pass

    # 5. Check training_curves.json
    tc_path = folder / "training_curves.json"
    if tc_path.exists():
        try:
            with open(tc_path, "r", encoding="utf-8") as f:
                tc_data = json.load(f)
                if tc_data.get("smoke_test"):
                    reasons.append("training_curves.json contains smoke test markings")
        except Exception:
            pass

    # 6. Check checkpoint files directly in folder
    for ckpt_name in ["last_checkpoint.pt", "best_checkpoint.pt"]:
        ckpt_path = folder / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if ckpt.get("is_smoke_test") or ckpt.get("config", {}).get("is_smoke_test"):
                    reasons.append(f"{ckpt_name} has is_smoke_test=True")
            except Exception:
                pass

    return reasons
