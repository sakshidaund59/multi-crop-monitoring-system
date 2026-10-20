#!/usr/bin/env python3
"""
Model Registry Registration Script for Multi-Crop Monitoring System (Phase 3A).
Registers trained EfficientNet-B0 classifiers in the model registry after validation.

Enforces Global Rule:
- No entries before real training exists.
- Checkpoint and metrics files must physically exist and be validated.
- Computes genuine SHA-256 checksum of model weights.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def register_trained_model(
    checkpoint_path: Path,
    metrics_path: Path,
    crop: str,
    model_name: Optional[str] = None,
    registry_file: Optional[Path] = None,
    notes: Optional[str] = None,
) -> Dict:
    """
    Register a verified trained model checkpoint in the model registry.
    Requires physical checkpoint and metrics report files.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot register model: Checkpoint not found at {checkpoint_path}. "
            "No registry entries are permitted before real training exists."
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot register model: Metrics report not found at {metrics_path}. "
            "Model must be evaluated on the held-out test split before registration."
        )

    # Compute weights checksum
    weights_sha256 = compute_sha256(checkpoint_path)

    # Read metrics JSON
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # Read checkpoint metadata
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Enforce rule: No smoke test outputs can ever be registered
    if ckpt.get("is_smoke_test") or ckpt.get("config", {}).get("is_smoke_test"):
        raise ValueError(
            "Cannot register model: Checkpoint is from a smoke test run ('SMOKE TEST, not valid metrics')."
        )

    # Resolve dataset version
    crop_lower = crop.lower()
    dataset_version = "unknown"
    meta_name_map = {
        "sugarcane": "sugarcane_sppu_metadata.json",
        "wheat": "wheat_zenodo_jic_metadata.json",
        "cotton": "cotton_mendeley_metadata.json",
        "rice": "rice_pest_lakthuru_groupsplit_v2_metadata.json",
    }
    meta_path = Path("ml_training/manifests") / meta_name_map.get(crop_lower, "")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                m_obj = json.load(mf)
                dataset_version = m_obj.get("dataset_version", "unknown")
        except Exception:
            pass

    resolved_name = model_name or f"{crop_lower}_disease_efficientnet_b0"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = {
        "model_name": resolved_name,
        "crop": crop_lower,
        "task": "classification",
        "architecture": "efficientnet_b0",
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "weights_sha256": weights_sha256,
        "dataset_version": dataset_version,
        "metrics_path": str(metrics_path).replace("\\", "/"),
        "accuracy": metrics_data.get("accuracy"),
        "macro_f1": metrics_data.get("macro_f1"),
        "weighted_f1": metrics_data.get("weighted_f1"),
        "total_test_samples": metrics_data.get("total_test_samples"),
        "best_epoch": ckpt.get("best_epoch", ckpt.get("epoch")),
        "registered_at": timestamp,
        "status": "candidate_for_production",
        "notes": notes or f"Phase 3A EfficientNet-B0 disease classifier for {crop_lower}.",
    }

    if crop_lower == "cotton":
        entry["notes"] += " Test split labelled indicative only due to tiny classes."

    # Load registry
    if registry_file is None:
        registry_file = Path("ml_training/registry/model_registry.json")
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    registry_entries: List[Dict] = []
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                registry_entries = json.load(f)
                if not isinstance(registry_entries, list):
                    registry_entries = []
        except Exception:
            registry_entries = []

    # Update or append
    updated = False
    for i, existing in enumerate(registry_entries):
        if existing.get("model_name") == resolved_name and existing.get("weights_sha256") == weights_sha256:
            registry_entries[i] = entry
            updated = True
            break
    if not updated:
        registry_entries.append(entry)

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry_entries, f, indent=2)

    print("\n" + "=" * 70)
    print(" MODEL REGISTRATION SUCCESSFUL")
    print("=" * 70)
    print(f"Model Name:        {entry['model_name']}")
    print(f"Crop:              {entry['crop']}")
    print(f"Weights SHA-256:   {entry['weights_sha256']}")
    print(f"Dataset Version:   {entry['dataset_version']}")
    print(f"Test Accuracy:     {entry['accuracy']*100:.2f}%" if entry['accuracy'] is not None else "N/A")
    print(f"Test Macro-F1:     {entry['macro_f1']*100:.2f}%" if entry['macro_f1'] is not None else "N/A")
    print(f"Registry Location: {registry_file}")
    print("=" * 70)

    return entry


def main():
    parser = argparse.ArgumentParser(description="Register a Verified Trained Classifier in the Model Registry.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--metrics", type=str, required=True,
                        help="Path to evaluation metrics JSON report.")
    parser.add_argument("--crop", type=str, required=True, choices=["sugarcane", "wheat", "cotton"],
                        help="Target crop.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Unique name for model (default: <crop>_disease_efficientnet_b0).")
    parser.add_argument("--registry-file", type=str, default="ml_training/registry/model_registry.json",
                        help="Path to model registry JSON file.")
    parser.add_argument("--notes", type=str, default=None,
                        help="Additional notes or provenance annotations.")

    args = parser.parse_args()
    register_trained_model(
        checkpoint_path=Path(args.checkpoint),
        metrics_path=Path(args.metrics),
        crop=args.crop,
        model_name=args.model_name,
        registry_file=Path(args.registry_file),
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
