"""
Unit & Integration Tests for Phase 3A Disease Classifiers Pipeline.
Tests manifest loader, relative path resolution, classification metrics,
Wilson 95% score confidence interval, CropDataset transforms, and model registry governance.

Strict Rules Enforced:
- All test fixtures use temporary solid synthetic images created within pytest tmp_path.
- Never touches or alters physical production datasets.
- No GPU required (CPU-only execution).
"""

import csv
import json
import math
from pathlib import Path
import pytest
from PIL import Image
import torch

from ml_training.scripts.classifier_utils import (
    CropDataset,
    compute_classification_metrics,
    compute_wilson_score_interval,
    detect_smoke_artifacts,
    get_class_weights,
    load_manifest_rows,
    resolve_image_path,
)
from ml_training.scripts.train_classifier import get_transforms, train_classifier
from ml_training.scripts.register_model import register_trained_model


def create_tiny_image(path: Path, size: tuple = (32, 32), color: tuple = (0, 128, 0)):
    """Generate a tiny solid-color JPEG image inside tmp_path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")
    return path


# ==============================================================================
# 1. WILSON SCORE CONFIDENCE INTERVAL TESTS
# ==============================================================================

def test_wilson_score_interval_formula_and_bounds():
    """Verify Wilson 95% score interval correctness across normal and edge cases."""
    # Edge case: n = 0
    ci_zero = compute_wilson_score_interval(0, 0, confidence=0.95)
    assert ci_zero == (0.0, 0.0), f"Expected (0.0, 0.0) for n=0, got {ci_zero}"

    # Edge case: k = 0, n = 10
    lower_0, upper_0 = compute_wilson_score_interval(0, 10, confidence=0.95)
    assert lower_0 == 0.0, f"Expected lower bound 0.0 for k=0, got {lower_0}"
    assert 0.0 < upper_0 < 0.5, f"Expected upper bound between 0 and 0.5 for 0/10, got {upper_0}"

    # Edge case: k = n, n = 10
    lower_10, upper_10 = compute_wilson_score_interval(10, 10, confidence=0.95)
    assert upper_10 == 1.0, f"Expected upper bound 1.0 for k=n, got {upper_10}"
    assert 0.5 < lower_10 < 1.0, f"Expected lower bound between 0.5 and 1.0 for 10/10, got {lower_10}"

    # Symmetric case: k = 5, n = 10 (p = 0.5)
    lower_5, upper_5 = compute_wilson_score_interval(5, 10, confidence=0.95)
    center = (lower_5 + upper_5) / 2.0
    assert abs(center - 0.5) < 0.02, f"Expected center ~0.5 for 5/10, got {center}"

    # Independent Wilson formula reference implementation
    def _independent_wilson_ref(k: int, n: int, z: float = 1.959963984540054) -> tuple:
        if n <= 0:
            return 0.0, 0.0
        k = max(0, min(k, n))
        p = k / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        half_width = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
        lower = max(0.0, center - half_width)
        upper = min(1.0, center + half_width)
        return round(lower, 4), round(upper, 4)

    # Reference case 1: k = 2, n = 10 (p = 0.2)
    # Expected interval is [0.0567, 0.5098] with tolerance 0.001
    l_ref, u_ref = compute_wilson_score_interval(2, 10, confidence=0.95)
    assert abs(l_ref - 0.0567) <= 0.001, f"Expected lower ~0.0567, got {l_ref}"
    assert abs(u_ref - 0.5098) <= 0.001, f"Expected upper ~0.5098, got {u_ref}"
    l_exp1, u_exp1 = _independent_wilson_ref(2, 10)
    assert abs(l_ref - l_exp1) <= 0.0001, f"Mismatch with independent Wilson formula: {l_ref} vs {l_exp1}"
    assert abs(u_ref - u_exp1) <= 0.0001, f"Mismatch with independent Wilson formula: {u_ref} vs {u_exp1}"

    # Reference case 2: k = 5, n = 10 (p = 0.5)
    # Expected interval is [0.2366, 0.7634] with tolerance 0.001
    l_5_10, u_5_10 = compute_wilson_score_interval(5, 10, confidence=0.95)
    assert abs(l_5_10 - 0.2366) <= 0.001, f"Expected lower ~0.2366, got {l_5_10}"
    assert abs(u_5_10 - 0.7634) <= 0.001, f"Expected upper ~0.7634, got {u_5_10}"
    l_exp2, u_exp2 = _independent_wilson_ref(5, 10)
    assert abs(l_5_10 - l_exp2) <= 0.0001, f"Mismatch with independent Wilson formula: {l_5_10} vs {l_exp2}"
    assert abs(u_5_10 - u_exp2) <= 0.0001, f"Mismatch with independent Wilson formula: {u_5_10} vs {u_exp2}"

    # Reference case 3: k = 0, n = 2 (p = 0.0)
    # Expected interval is [0.0, 0.6576] with tolerance 0.001
    l_0_2, u_0_2 = compute_wilson_score_interval(0, 2, confidence=0.95)
    assert abs(l_0_2 - 0.0) <= 0.001, f"Expected lower ~0.0, got {l_0_2}"
    assert abs(u_0_2 - 0.6576) <= 0.001, f"Expected upper ~0.6576, got {u_0_2}"
    l_exp3, u_exp3 = _independent_wilson_ref(0, 2)
    assert abs(l_0_2 - l_exp3) <= 0.0001, f"Mismatch with independent Wilson formula: {l_0_2} vs {l_exp3}"
    assert abs(u_0_2 - u_exp3) <= 0.0001, f"Mismatch with independent Wilson formula: {u_0_2} vs {u_exp3}"

    # Invariant: for all k in 0..n, 0 <= lower <= upper <= 1
    for n in [1, 2, 5, 20, 100]:
        for k in range(n + 1):
            lo, hi = compute_wilson_score_interval(k, n, confidence=0.95)
            assert 0.0 <= lo <= hi <= 1.0, f"Bounds violated for k={k}, n={n}: ({lo}, {hi})"


# ==============================================================================
# 2. CLASSIFICATION METRICS COMPUTATION TESTS
# ==============================================================================

def test_classification_metrics_computation():
    """Verify accuracy, per-class precision/recall/F1, and confusion matrix."""
    class_names = ["class_A", "class_B", "class_C"]
    # True classes:
    # 0, 0 (2 samples of A)
    # 1, 1, 1 (3 samples of B)
    # 2 (1 sample of C)
    y_true = [0, 0, 1, 1, 1, 2]
    # Predictions:
    # A -> 0 (correct), 1 (confused with B)
    # B -> 1, 1, 1 (all correct)
    # C -> 2 (correct)
    y_pred = [0, 1, 1, 1, 1, 2]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    # Accuracy: 5 correct out of 6 -> ~0.8333
    assert abs(metrics["accuracy"] - 0.8333) < 0.01, f"Expected accuracy ~0.8333, got {metrics['accuracy']}"
    assert metrics["total_samples"] == 6

    # Class A: TP=1, FN=1, FP=0 -> Recall=0.5, Precision=1.0, Support=2
    ca = metrics["per_class"]["class_A"]
    assert ca["support"] == 2
    assert ca["tp"] == 1
    assert ca["fn"] == 1
    assert ca["precision"] == 1.0
    assert ca["recall"] == 0.5
    assert len(ca["recall_wilson_95_ci"]) == 2
    assert ca["recall_wilson_95_ci"][0] <= ca["recall"] <= ca["recall_wilson_95_ci"][1]

    # Class B: TP=3, FN=0, FP=1 -> Recall=1.0, Precision=3/4=0.75, Support=3
    cb = metrics["per_class"]["class_B"]
    assert cb["support"] == 3
    assert cb["tp"] == 3
    assert cb["precision"] == 0.75
    assert cb["recall"] == 1.0

    # Macro-F1 should be average of the 3 class F1s
    expected_macro_f1 = (ca["f1"] + cb["f1"] + metrics["per_class"]["class_C"]["f1"]) / 3.0
    assert abs(metrics["macro_f1"] - expected_macro_f1) < 0.01

    # Confusion Matrix checks
    cm = metrics["confusion_matrix"]
    assert len(cm) == 3 and len(cm[0]) == 3
    assert cm[0][0] == 1 and cm[0][1] == 1  # True A predicted A, True A predicted B
    assert cm[1][1] == 3                     # True B predicted B
    assert cm[2][2] == 1                     # True C predicted C


# ==============================================================================
# 3. MANIFEST LOADER & RELATIVE PATH TESTS
# ==============================================================================

def test_manifest_loader_relative_path_and_normalization(tmp_path):
    """Test manifest loader normalises Windows backslashes and filters splits."""
    manifest_csv = tmp_path / "test_manifest.csv"
    fieldnames = ["path", "class", "split", "sha256", "group_id"]
    rows = [
        {"path": "folder\\sub\\img1.jpg", "class": "Healthy", "split": "train", "sha256": "aaa", "group_id": "g1"},
        {"path": "folder/sub/img2.jpg", "class": "Rust", "split": "val", "sha256": "bbb", "group_id": "g2"},
        {"path": "folder\\sub\\img3.jpg", "class": "RedRot", "split": "test", "sha256": "ccc", "group_id": "g3"},
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 1. Load all rows without split filter
    loaded_all = load_manifest_rows(manifest_csv, crop="wheat")
    assert len(loaded_all) == 3
    # Check normalisation
    assert loaded_all[0]["path"] == "folder/sub/img1.jpg"
    assert "\\" not in loaded_all[0]["path"]

    # 2. Split filtering
    tr_rows = load_manifest_rows(manifest_csv, crop="wheat", split="train")
    assert len(tr_rows) == 1
    assert tr_rows[0]["path"] == "folder/sub/img1.jpg"

    te_rows = load_manifest_rows(manifest_csv, crop="wheat", split="test")
    assert len(te_rows) == 1
    assert te_rows[0]["path"] == "folder/sub/img3.jpg"


def test_sugarcane_use_for_eval_filter(tmp_path):
    """Test sugarcane deduplication filtering on val and test splits."""
    manifest_csv = tmp_path / "sugarcane_manifest.csv"
    fieldnames = ["path", "class", "split", "sha256", "group_id", "use_for_eval"]
    rows = [
        {"path": "img_tr1.jpg", "class": "Healthy", "split": "train", "sha256": "1", "group_id": "g1", "use_for_eval": "true"},
        {"path": "img_tr2.jpg", "class": "Healthy", "split": "train", "sha256": "2", "group_id": "g1", "use_for_eval": "true"},
        {"path": "img_val_rep.jpg", "class": "Rust", "split": "val", "sha256": "3", "group_id": "g2", "use_for_eval": "true"},
        {"path": "img_val_dup.jpg", "class": "Rust", "split": "val", "sha256": "4", "group_id": "g2", "use_for_eval": "false"},
        {"path": "img_te_rep.jpg", "class": "Yellow", "split": "test", "sha256": "5", "group_id": "g3", "use_for_eval": "true"},
        {"path": "img_te_dup.jpg", "class": "Yellow", "split": "test", "sha256": "6", "group_id": "g3", "use_for_eval": "false"},
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Train should keep all 2 samples
    train_loaded = load_manifest_rows(manifest_csv, crop="sugarcane", split="train", filter_eval_only=False)
    assert len(train_loaded) == 2

    # Val with filter_eval_only=True should keep only the 1 deduplicated representative
    val_filtered = load_manifest_rows(manifest_csv, crop="sugarcane", split="val", filter_eval_only=True)
    assert len(val_filtered) == 1
    assert val_filtered[0]["path"] == "img_val_rep.jpg"

    # Test with filter_eval_only=True should keep only the 1 deduplicated representative
    test_filtered = load_manifest_rows(manifest_csv, crop="sugarcane", split="test", filter_eval_only=True)
    assert len(test_filtered) == 1
    assert test_filtered[0]["path"] == "img_te_rep.jpg"


# ==============================================================================
# 4. RESOLVE IMAGE PATH HIERARCHIES TESTS
# ==============================================================================

def test_resolve_image_path_hierarchies(tmp_path):
    """Verify path resolution finds images under dataset_name, crop, or direct root."""
    data_root = tmp_path / "data"

    # Case A: data_root / dataset_name / rel_path (e.g. data_root / sugarcane_sppu / Healthy / 1.jpg)
    img_a = create_tiny_image(data_root / "sugarcane_sppu" / "Healthy" / "1.jpg")
    resolved_a = resolve_image_path(data_root, "sugarcane", "Healthy\\1.jpg")
    assert resolved_a.resolve() == img_a.resolve()

    # Case B: data_root / crop / rel_path (e.g. data_root / wheat / Mildew / 2.jpg)
    img_b = create_tiny_image(data_root / "wheat" / "Mildew" / "2.jpg")
    resolved_b = resolve_image_path(data_root, "wheat", "Mildew/2.jpg")
    assert resolved_b.resolve() == img_b.resolve()

    # Case C: data_root / rel_path (direct root when data_root points to dataset folder)
    img_c = create_tiny_image(data_root / "bacterial blight" / "3.jpg")
    resolved_c = resolve_image_path(data_root, "cotton", "bacterial blight/3.jpg")
    assert resolved_c.resolve() == img_c.resolve()


# ==============================================================================
# 5. COTTON CLASS WEIGHTS EXTRACTION TEST
# ==============================================================================

def test_cotton_class_weights_extraction(tmp_path):
    """Verify class weights are properly parsed and mapped to index order."""
    meta_path = tmp_path / "cotton_metadata.json"
    meta_data = {
        "dataset_name": "cotton_mendeley",
        "train_class_weights": {
            "bacterial blight": 2.0663,
            "curl virus": 9.5278,
            "fussarium wilt": 5.1970,
            "healthy": 0.3107,
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    class_to_idx = {
        "bacterial blight": 0,
        "curl virus": 1,
        "fussarium wilt": 2,
        "healthy": 3,
    }

    weights = get_class_weights(meta_path, class_to_idx)
    assert isinstance(weights, torch.Tensor)
    assert len(weights) == 4
    assert abs(weights[0].item() - 2.0663) < 1e-4
    assert abs(weights[1].item() - 9.5278) < 1e-4
    assert abs(weights[2].item() - 5.1970) < 1e-4
    assert abs(weights[3].item() - 0.3107) < 1e-4


# ==============================================================================
# 6. PYTORCH DATASET & TRANSFORMS TEST
# ==============================================================================

def test_crop_dataset_and_transforms(tmp_path):
    """Verify CropDataset loads PIL image and applies transforms producing (3, 224, 224)."""
    img_path = create_tiny_image(tmp_path / "test.jpg", size=(64, 64))
    train_tf, eval_tf = get_transforms(img_size=224)

    samples = [(img_path, 1)]
    ds_train = CropDataset(samples, transform=train_tf)
    ds_eval = CropDataset(samples, transform=eval_tf)

    assert len(ds_train) == 1
    t_img, t_lbl = ds_train[0]
    assert isinstance(t_img, torch.Tensor)
    assert t_img.shape == (3, 224, 224)
    assert t_lbl == 1

    e_img, e_lbl = ds_eval[0]
    assert isinstance(e_img, torch.Tensor)
    assert e_img.shape == (3, 224, 224)
    assert e_lbl == 1


# ==============================================================================
# 7. MODEL REGISTRY GOVERNANCE TEST
# ==============================================================================

def test_register_model_requires_physical_files(tmp_path):
    """Enforce rule: No registry entries before real physical checkpoint and metrics exist."""
    fake_ckpt = tmp_path / "does_not_exist.pt"
    fake_metrics = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        register_trained_model(
            checkpoint_path=fake_ckpt,
            metrics_path=fake_metrics,
            crop="sugarcane",
            registry_file=tmp_path / "model_registry.json",
        )
    assert "Checkpoint not found" in str(exc_info.value)


def test_register_model_rejects_smoke_checkpoint(tmp_path):
    """Enforce rule: Checkpoints from smoke test runs cannot be registered."""
    smoke_ckpt = tmp_path / "smoke_checkpoint.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": {},
        "is_smoke_test": True,
        "smoke_test": "SMOKE TEST, not valid metrics",
    }, smoke_ckpt)

    metrics_file = tmp_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": 0.5, "macro_f1": 0.5}, f)

    with pytest.raises(ValueError) as exc_info:
        register_trained_model(
            checkpoint_path=smoke_ckpt,
            metrics_path=metrics_file,
            crop="sugarcane",
            registry_file=tmp_path / "model_registry.json",
        )
    assert "smoke test" in str(exc_info.value).lower()


# ==============================================================================
# 8. SMOKE TEST SAFETY & GOVERNANCE TESTS
# ==============================================================================

def test_resume_refuses_smoke_checkpoint(tmp_path):
    """Enforce rule: --resume must REFUSE (raise ValueError) if checkpoint has is_smoke_test = true."""
    smoke_dir = tmp_path / "smoke_dir"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_ckpt = smoke_dir / "last_checkpoint.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "is_smoke_test": True,
        "smoke_test": "SMOKE TEST, not valid metrics",
    }, smoke_ckpt)

    class MockArgs:
        crop = "sugarcane"
        manifest = None
        data_root = None
        out_dir = str(smoke_dir)
        epochs = 1
        batch_size = 2
        lr = 1e-3
        seed = 42
        img_size = 224
        device = "cpu"
        resume = True
        smoke = True

    with pytest.raises(ValueError) as exc_info:
        train_classifier(MockArgs())
    err_text = str(exc_info.value).lower()
    assert "cannot resume" in err_text or "is_smoke_test" in err_text or "smoke" in err_text


def test_real_run_refuses_folder_with_smoke_artifacts(tmp_path):
    """Enforce rule: A real run (smoke=False) must refuse to start in a folder containing smoke artifacts."""
    run_dir = tmp_path / "real_run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Place a smoke artifact (e.g. config.json marked as smoke test)
    cfg_file = run_dir / "config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump({"is_smoke_test": True, "smoke_test": "SMOKE TEST, not valid metrics"}, f)

    class MockArgs:
        crop = "sugarcane"
        manifest = None
        data_root = None
        out_dir = str(run_dir)
        epochs = 30
        batch_size = 32
        lr = 5e-4
        seed = 42
        img_size = 224
        device = "cpu"
        resume = False
        smoke = False

    with pytest.raises(ValueError) as exc_info:
        train_classifier(MockArgs())
    assert "smoke artifacts" in str(exc_info.value).lower()


def test_detect_smoke_artifacts_detection_rules(tmp_path):
    """Verify detect_smoke_artifacts detects smoke folder names, subdirectories, and files."""
    # 1. Clean folder
    clean_dir = tmp_path / "clean_model_dir"
    clean_dir.mkdir(parents=True, exist_ok=True)
    assert detect_smoke_artifacts(clean_dir) == []

    # 2. Folder with smoke subdirectory
    dir_with_sub = tmp_path / "dir_with_sub"
    dir_with_sub.mkdir(parents=True, exist_ok=True)
    (dir_with_sub / "_smoke_sugarcane").mkdir(parents=True, exist_ok=True)
    reasons = detect_smoke_artifacts(dir_with_sub)
    assert any("smoke subdirectory" in r.lower() for r in reasons)

    # 3. Folder named _smoke_<crop>
    smoke_named_dir = tmp_path / "_smoke_sugarcane"
    smoke_named_dir.mkdir(parents=True, exist_ok=True)
    reasons = detect_smoke_artifacts(smoke_named_dir)
    assert any("directory name" in r.lower() for r in reasons)

    # 4. Folder with smoke checkpoint
    ckpt_dir = tmp_path / "ckpt_dir"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"is_smoke_test": True}, ckpt_dir / "last_checkpoint.pt")
    reasons = detect_smoke_artifacts(ckpt_dir)
    assert any("last_checkpoint.pt" in r for r in reasons)
