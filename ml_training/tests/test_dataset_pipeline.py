"""
Unit & Integration Tests for Phase 2 Dataset Pipeline.
Tests use temporary solid-color images created in-memory strictly within the test
fixtures using pytest's tmp_path. Never touches application or production data.
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from PIL import Image

from ml_training.scripts.validate_dataset import (
    detect_layout,
    validate_dataset,
    validate_classification,
    validate_yolo,
    validate_voc,
    compute_sha256,
)
from ml_training.scripts.build_splits import build_splits, cluster_into_groups
from ml_training.scripts.dataset_report import generate_dataset_report, parse_manifest_csv


def create_solid_image(path: Path, size: tuple = (64, 64), color: tuple = (255, 0, 0)):
    """Helper to generate a real, valid JPEG image inside temporary test directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")
    return path


def test_missing_or_empty_dataset(tmp_path):
    """Test that missing or empty dataset folders are cleanly detected without crashing."""
    # 1. Non-existent path
    non_existent = tmp_path / "does_not_exist"
    report = validate_dataset(str(non_existent))
    assert "error" in report
    assert "missing" in report["error"].lower()

    # 2. Empty directory
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    layout, _ = detect_layout(empty_dir)
    assert layout == "empty"

    report2 = validate_dataset(str(empty_dir))
    assert "error" in report2
    assert "empty" in report2["error"].lower()


def test_classification_validation(tmp_path):
    """Test validation of classification layout: valid, corrupt, tiny images."""
    ds_dir = tmp_path / "rice_test"
    class_blast = ds_dir / "blast"
    class_brown = ds_dir / "brown_spot"

    # Create 2 valid images in blast
    create_solid_image(class_blast / "img1.jpg", size=(64, 64), color=(100, 150, 200))
    create_solid_image(class_blast / "img2.jpg", size=(64, 64), color=(50, 100, 150))

    # Create 1 valid image in brown_spot
    create_solid_image(class_brown / "img3.jpg", size=(64, 64), color=(10, 20, 30))

    # Create 1 tiny image (< 32px) in brown_spot
    create_solid_image(class_brown / "tiny.jpg", size=(16, 16), color=(20, 30, 40))

    # Create 1 corrupt image
    corrupt_path = class_brown / "corrupt.jpg"
    corrupt_path.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CORRUPT_BYTES_12345")

    layout, meta = detect_layout(ds_dir)
    assert layout == "classification"
    assert "blast" in meta["class_dirs"]
    assert "brown_spot" in meta["class_dirs"]

    report_dir = tmp_path / "reports"
    report = validate_classification(ds_dir, min_size=32, hash_dist_threshold=2)

    assert report["total_images"] == 5
    assert report["valid_images"] == 3
    assert len(report["tiny_images"]) == 1
    assert "tiny.jpg" in report["tiny_images"][0]["path"]
    assert len(report["corrupt_images"]) == 1
    assert "corrupt.jpg" in report["corrupt_images"][0]["path"]
    assert report["classes"]["blast"] == 2
    assert report["classes"]["brown_spot"] == 1


def test_yolo_validation(tmp_path):
    """Test validation of YOLO layout: missing labels, empty labels, out-of-range bboxes."""
    ds_dir = tmp_path / "yolo_test"
    images_dir = ds_dir / "images"
    labels_dir = ds_dir / "labels"

    # 1. Valid image and label
    create_solid_image(images_dir / "scan1.jpg", size=(100, 100))
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "scan1.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    # 2. Image with out-of-range coordinates (> 1.0)
    create_solid_image(images_dir / "scan2.jpg", size=(100, 100))
    (labels_dir / "scan2.txt").write_text("1 1.25 0.5 0.4 0.4\n", encoding="utf-8")

    # 3. Image with missing label
    create_solid_image(images_dir / "scan3.jpg", size=(100, 100))

    # 4. Image with empty label
    create_solid_image(images_dir / "scan4.jpg", size=(100, 100))
    (labels_dir / "scan4.txt").write_text("", encoding="utf-8")

    layout, _ = detect_layout(ds_dir)
    assert layout == "yolo"

    report = validate_yolo(ds_dir, min_size=32, hash_dist_threshold=2)
    assert report["total_images"] == 4
    assert len(report["missing_labels"]) == 1
    assert "scan3.jpg" in report["missing_labels"][0]
    assert len(report["empty_labels"]) == 1
    assert "scan4.txt" in report["empty_labels"][0]
    assert len(report["invalid_bboxes"]) == 1
    assert "scan2.txt" in report["invalid_bboxes"][0]["label_path"]


def test_voc_xml_validation(tmp_path):
    """Test validation of Pascal VOC XML layout with bboxes and coordinates."""
    ds_dir = tmp_path / "voc_test"
    jpeg_dir = ds_dir / "JPEGImages"
    annot_dir = ds_dir / "Annotations"

    # 1. Valid image and VOC XML
    create_solid_image(jpeg_dir / "field1.jpg", size=(200, 100))
    annot_dir.mkdir(parents=True, exist_ok=True)

    xml_valid = """<annotation>
        <size><width>200</width><height>100</height></size>
        <object>
            <name>armyworm</name>
            <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
        </object>
    </annotation>"""
    (annot_dir / "field1.xml").write_text(xml_valid, encoding="utf-8")

    # 2. Image with invalid bbox (xmax > width 200)
    create_solid_image(jpeg_dir / "field2.jpg", size=(200, 100))
    xml_invalid = """<annotation>
        <size><width>200</width><height>100</height></size>
        <object>
            <name>armyworm</name>
            <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>250</xmax><ymax>50</ymax></bndbox>
        </object>
    </annotation>"""
    (annot_dir / "field2.xml").write_text(xml_invalid, encoding="utf-8")

    layout, _ = detect_layout(ds_dir)
    assert layout == "voc"

    report = validate_voc(ds_dir, min_size=32, hash_dist_threshold=2)
    assert report["total_images"] == 2
    assert report["valid_images"] == 2
    assert len(report["invalid_bboxes"]) == 1
    assert "field2.xml" in report["invalid_bboxes"][0]["xml_path"]
    assert "out of image bounds" in report["invalid_bboxes"][0]["reason"]


def test_build_splits_group_aware(tmp_path):
    """Test group-aware 70/20/10 split generation and manifest CSV hashing."""
    ds_dir = tmp_path / "raw" / "cotton_test"
    c_healthy = ds_dir / "healthy"
    c_bollworm = ds_dir / "bollworm"

    # Create images with plant prefixes (plant1, plant2, plant3)
    # plant1 burst shots:
    create_solid_image(c_healthy / "plant01_01.jpg", color=(10, 10, 10))
    create_solid_image(c_healthy / "plant01_02.jpg", color=(10, 10, 10))
    create_solid_image(c_healthy / "plant02_01.jpg", color=(20, 20, 20))
    create_solid_image(c_healthy / "plant02_02.jpg", color=(20, 20, 20))
    create_solid_image(c_healthy / "plant03_01.jpg", color=(30, 30, 30))

    # bollworm shots:
    create_solid_image(c_bollworm / "plant04_01.jpg", color=(40, 40, 40))
    create_solid_image(c_bollworm / "plant04_02.jpg", color=(40, 40, 40))
    create_solid_image(c_bollworm / "plant05_01.jpg", color=(50, 50, 50))
    create_solid_image(c_bollworm / "plant05_02.jpg", color=(50, 50, 50))
    create_solid_image(c_bollworm / "plant06_01.jpg", color=(60, 60, 60))

    manifests_dir = tmp_path / "manifests"

    stats = build_splits(
        dataset_name="cotton_test",
        raw_dir=tmp_path / "raw",
        manifests_dir=manifests_dir,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
    )

    assert "error" not in stats
    assert stats["total_images"] == 10
    assert len(stats["dataset_version"]) == 12

    # Check manifest CSV
    manifest_csv = manifests_dir / "cotton_test_manifest.csv"
    assert manifest_csv.exists()

    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    assert len(rows) == 10
    for r in rows:
        assert r["split"] in ["train", "val", "test"]
        assert len(r["sha256"]) == 64
        assert r["group_id"].startswith("grp_")

    # CRITICAL CHECK: Verify that items in the same group are in the EXACT SAME split
    group_to_splits = {}
    for r in rows:
        grp = r["group_id"]
        sp = r["split"]
        if grp not in group_to_splits:
            group_to_splits[grp] = sp
        else:
            assert group_to_splits[grp] == sp, f"Group {grp} leaked across splits: {group_to_splits[grp]} vs {sp}"


def test_dataset_report_generator(tmp_path):
    """Test generating docs/dataset.md with real statistics from manifests."""
    manifests_dir = tmp_path / "manifests"
    raw_dir = tmp_path / "raw"
    sources_yaml = tmp_path / "dataset_sources.yaml"
    output_md = tmp_path / "docs" / "dataset.md"

    # Setup cotton dataset
    c_dir = raw_dir / "maize_blight" / "blight"
    create_solid_image(c_dir / "img1.jpg", size=(128, 128))
    create_solid_image(c_dir / "img2.jpg", size=(256, 256))

    build_splits(
        dataset_name="maize_blight",
        raw_dir=raw_dir,
        manifests_dir=manifests_dir,
    )

    # Populate sources
    sources_data = {
        "datasets": {
            "maize_blight": {
                "crop": "maize",
                "source_name": "CIMMYT Maize Callset",
                "source_url": "https://example.org/cimmyt",
                "license": "CC BY 4.0",
                "citation": "CIMMYT Research Team, 2024",
            }
        }
    }
    with open(sources_yaml, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(sources_data, f)

    content = generate_dataset_report(
        manifests_dir=manifests_dir,
        raw_dir=raw_dir,
        sources_path=sources_yaml,
        output_path=output_md,
    )

    assert output_md.exists()
    assert "maize_blight" in content
    assert "Maize" in content
    assert "CC BY 4.0" in content
    assert "128x128" in content
    assert "256x256" in content


def test_normalise_source_id_canonical():
    """Unit test normalise_source_id with exact user-specified canonical edge cases."""
    from ml_training.scripts.audit_and_harden_splits import normalise_source_id

    cases = [
        ("sborer-1-jpeg_11111111-2222-3333-4444-555555555555_jpeg_jpg.rf.abcdef1234567890.jpg", "sborer-1"),
        ("sborer-1-jpeg_22222222-3333-4444-5555-666666666666_jpeg_jpg.rf.0987654321fedcba.jpg", "sborer-1"),
        ("sborer-1-_jpeg_jpg.rf.abcdef1234567890.jpg", "sborer-1"),
        ("STEM-BORER_original_sborer-33-jpg_33333333-4444-5555-6666-777777777777.jpg", "sborer-33"),
        ("69541_jpg.rf.c6d12999cb51fa30a0bf05b63aa5391e.jpg", "69541"),
        ("GREEN-LEAFHOPPERS_original_69541-jpg_10b0e515-568b-49e0-b6c8-5081f2115e51_jpg.rf.c7b1aa258410b0a2bf04fb54fcfa2613.jpg", "69541"),
        ("BROWN-PLANTHOPPER_original_bhopper-107-jpg_5aea44b9-6712-471b-85e4-38a4ef3d74f2_jpg.rf.d440726058ac5729bc04e27e3fc3d1a3.jpg", "bhopper-107"),
        ("LEAF-FOLDERS_original_lfolder-61-jpg_fd683a07-f809-41f6-8a89-8f6a8f42038a_jpg.rf.048577e650f8d74eca42655cd89de6f5.jpg", "lfolder-61"),
        ("WHORL-MAGGOTS_original_whorl-163-jpg_3615f528-3b1a-4e6f-a446-b2be4e748aab_jpg.rf.710ab760439253ef02eece7ee16ecabd.jpg", "whorl-163"),
        ("bhopper-1-png", "bhopper-1"),
        ("sborer-2-png", "sborer-2"),
        ("bhopper-1_png", "bhopper-1"),
        ("sborer-1-png", "sborer-1"),
        ("bhopper-1-jpeg", "bhopper-1"),
        ("BROWN-PLANTHOPPER_original_bhopper-1-png_5aea44b9-6712-471b-85e4-38a4ef3d74f2_jpg.rf.d440726058ac5729bc04e27e3fc3d1a3.jpg", "bhopper-1"),
        ("STEM-BORER_original_sborer-2-png_317b597e-bca1-42fe-8560-bc3dfea292da_jpg.rf.ea0c3c9de86cb2b14f434d7278e62e1f.jpg", "sborer-2"),
    ]
    for fn, expected in cases:
        got = normalise_source_id(fn)
        assert got == expected, f"normalise_source_id('{fn}') -> '{got}', expected '{expected}'"


def test_rice_pest_groupsplit_leakage_zero():
    """Automated Leakage Test: Verify 0 exact and near-duplicate cross-split pairs on v2 manifest."""
    from collections import defaultdict
    from ml_training.scripts.audit_and_harden_splits import normalise_source_id

    manifest_v2 = Path("ml_training/manifests/rice_pest_lakthuru_groupsplit_v2_manifest.csv")
    manifest_v1 = Path("ml_training/manifests/rice_pest_lakthuru_groupsplit_manifest.csv")
    manifest_path = manifest_v2 if manifest_v2.exists() else manifest_v1

    if not manifest_path.exists():
        pytest.skip("rice pest group split manifest not yet generated")

    sha_to_splits = defaultdict(set)
    group_to_splits = defaultdict(set)
    source_id_to_splits = defaultdict(set)

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        for r in reader:
            sp = r["split"]
            sha_to_splits[r["sha256"]].add(sp)
            group_to_splits[r["group_id"]].add(sp)
            s_id = normalise_source_id(r["path"])
            source_id_to_splits[s_id].add(sp)

    # 1. Exact duplicates must not cross splits
    cross_split_sha = {sha: sps for sha, sps in sha_to_splits.items() if len(sps) > 1}
    assert len(cross_split_sha) == 0, f"Found {len(cross_split_sha)} exact duplicate cross-split leaks!"

    # 2. Groups must not cross splits
    cross_split_grp = {grp: sps for grp, sps in group_to_splits.items() if len(sps) > 1}
    assert len(cross_split_grp) == 0, f"Found {len(cross_split_grp)} group cross-split leaks!"

    # 3. Source IDs must not cross splits in v2
    if manifest_path == manifest_v2:
        cross_split_src = {sid: sps for sid, sps in source_id_to_splits.items() if len(sps) > 1}
        assert len(cross_split_src) == 0, f"Found {len(cross_split_src)} source ID cross-split leaks in v2: {cross_split_src}!"


def test_sugarcane_use_for_eval_flag():
    """Test that sugarcane manifest includes use_for_eval column and single representation in val/test."""
    from collections import defaultdict
    manifest_path = Path("ml_training/manifests/sugarcane_sppu_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("sugarcane_sppu_manifest.csv does not exist")

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        assert "use_for_eval" in reader.fieldnames, "use_for_eval column missing from sugarcane manifest"

        eval_by_split_group = defaultdict(int)
        for r in reader:
            sp = r["split"]
            grp = r["group_id"]
            is_eval = r["use_for_eval"].lower() == "true"
            if sp in ("val", "test") and is_eval:
                eval_by_split_group[(sp, grp)] += 1

        # In val and test, at most 1 representative image per duplicate group should be used for eval
        for (sp, grp), count in eval_by_split_group.items():
            assert count == 1, f"Group {grp} in {sp} has {count} eval images, expected exactly 1!"


def test_cotton_disease_manifest():
    """Test that cotton manifest maps 4 disease classes and retains severity metadata."""
    manifest_path = Path("ml_training/manifests/cotton_mendeley_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("cotton_mendeley_manifest.csv does not exist")

    valid_diseases = {"healthy", "bacterial blight", "fussarium wilt", "curl virus"}
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        assert "severity" in reader.fieldnames, "severity metadata column missing from cotton manifest"
        for r in reader:
            assert r["class"] in valid_diseases, f"Unexpected cotton disease class: {r['class']}"
            assert r["severity"] in ("critical", "mild", "moderate", "none"), f"Unexpected severity: {r['severity']}"


def test_split_reproducibility(tmp_path):
    """Verify that running the split builder twice produces byte-identical manifests and versions."""
    from ml_training.scripts.audit_and_harden_splits import (
        build_reproducible_cotton_manifest,
        get_data_root,
    )

    raw_dir = get_data_root()
    if not (raw_dir / "cotton_mendeley").exists():
        pytest.skip("cotton_mendeley raw dataset not found")

    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    res1 = build_reproducible_cotton_manifest(raw_dir, dir1, seed=42)
    res2 = build_reproducible_cotton_manifest(raw_dir, dir2, seed=42)

    assert res1["version"] == res2["version"]
    f1 = (dir1 / "cotton_mendeley_manifest.csv").read_bytes()
    f2 = (dir2 / "cotton_mendeley_manifest.csv").read_bytes()
    assert f1 == f2, "Manifests between consecutive runs differ!"


