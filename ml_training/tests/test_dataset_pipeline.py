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
