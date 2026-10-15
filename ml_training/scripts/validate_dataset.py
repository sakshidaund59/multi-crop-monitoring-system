#!/usr/bin/env python3
"""
Dataset Validator for Multi-Crop Monitoring System (Phase 2).
Validates agricultural datasets for classification, YOLO detection, and Pascal VOC XML layouts.
Detects corrupt images, tiny images, missing/empty labels, out-of-range bboxes,
exact duplicates, and cross-class near-duplicates using perceptual hashing.

All rules enforced: No fake data, no synthesized labels or images.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def compute_sha256(file_path: Path) -> str:
    """Compute standard SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_layout(dataset_dir: Path) -> Tuple[str, Dict]:
    """
    Detect dataset layout:
    - 'classification': folders per class containing images
    - 'yolo': 'images/' and 'labels/' directories (or data.yaml)
    - 'voc': 'Annotations/' and 'JPEGImages/' (or XMLs present with images)
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return "missing", {}

    entries = list(dataset_dir.iterdir())
    if not entries:
        return "empty", {}

    subdirs = [p.name.lower() for p in entries if p.is_dir()]
    files = [p.name.lower() for p in entries if p.is_file()]

    # 1. Check for YOLO layout
    has_yolo_yaml = any(f.endswith(".yaml") or f.endswith(".yml") for f in files)
    has_images_dir = "images" in subdirs
    has_labels_dir = "labels" in subdirs
    has_split_dirs = any(s in subdirs for s in ["train", "val", "test"])

    if has_images_dir and has_labels_dir:
        return "yolo", {"type": "standard_yolo"}
    if has_yolo_yaml and (has_images_dir or has_split_dirs):
        return "yolo", {"type": "configured_yolo"}
    if has_split_dirs:
        # Check inside train/
        train_dir = dataset_dir / "train"
        if train_dir.is_dir():
            train_subdirs = [p.name.lower() for p in train_dir.iterdir() if p.is_dir()]
            if "images" in train_subdirs and "labels" in train_subdirs:
                return "yolo", {"type": "split_yolo"}

    # 2. Check for Pascal VOC layout
    if "annotations" in subdirs and ("jpegimages" in subdirs or "images" in subdirs):
        return "voc", {"type": "standard_voc"}

    xml_files = list(dataset_dir.glob("*.xml"))
    img_files = [p for p in entries if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    if len(xml_files) > 0 and len(img_files) > 0:
        return "voc", {"type": "flat_voc"}

    # 3. Check for Classification layout (folders per class containing images)
    class_dirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    if class_dirs:
        # Check if subdirs contain images
        valid_class_dirs = 0
        for cd in class_dirs:
            if any(p.suffix.lower() in SUPPORTED_IMAGE_EXTS for p in cd.rglob("*")):
                valid_class_dirs += 1
        if valid_class_dirs > 0:
            return "classification", {"class_dirs": [d.name for d in class_dirs]}

    return "unknown", {}


def validate_single_image(
    image_path: Path, min_size: int = 32
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str], Optional[str]]:
    """
    Validates an image file:
    Returns (is_valid, error_msg, (width, height), sha256_hash, phash_str)
    """
    try:
        sha256 = compute_sha256(image_path)
    except Exception as e:
        return False, f"File unreadable: {str(e)}", None, None, None

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        return False, f"Corrupt image file (verify failed): {str(e)}", None, sha256, None

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < min_size or height < min_size:
                return (
                    False,
                    f"Tiny image: dimensions ({width}x{height}) below minimum {min_size}px",
                    (width, height),
                    sha256,
                    None,
                )
            
            # Compute perceptual hash if imagehash is available
            phash_str = None
            if HAS_IMAGEHASH:
                try:
                    phash_str = str(imagehash.phash(img))
                except Exception:
                    phash_str = None

            return True, None, (width, height), sha256, phash_str
    except Exception as e:
        return False, f"Cannot decode image pixels: {str(e)}", None, sha256, None


def validate_classification(
    dataset_dir: Path, min_size: int, hash_dist_threshold: int
) -> Dict:
    """Validate a classification dataset with folder-per-class layout."""
    report = {
        "layout": "classification",
        "dataset_path": str(dataset_dir),
        "total_images": 0,
        "valid_images": 0,
        "classes": {},
        "corrupt_images": [],
        "tiny_images": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "cross_class_near_duplicates": [],
        "summary": {},
    }

    class_dirs = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ["__pycache__"]
    ]

    sha256_map: Dict[str, List[Dict]] = defaultdict(list)
    phash_list: List[Dict] = []
    classes_counts = defaultdict(int)

    for c_dir in sorted(class_dirs, key=lambda x: x.name):
        class_name = c_dir.name
        img_paths = [
            p for p in c_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]

        for img_p in img_paths:
            report["total_images"] += 1
            rel_path = str(img_p.relative_to(dataset_dir))

            is_valid, err, dims, sha256, phash = validate_single_image(img_p, min_size=min_size)

            if not is_valid:
                if "Tiny image" in (err or ""):
                    report["tiny_images"].append({"path": rel_path, "class": class_name, "reason": err, "dimensions": dims})
                else:
                    report["corrupt_images"].append({"path": rel_path, "class": class_name, "reason": err})
                continue

            report["valid_images"] += 1
            classes_counts[class_name] += 1

            record = {
                "path": rel_path,
                "class": class_name,
                "sha256": sha256,
                "phash": phash,
                "dimensions": dims,
            }
            sha256_map[sha256].append(record)
            if phash:
                phash_list.append(record)

    report["classes"] = dict(classes_counts)

    # Detect exact duplicates (identical SHA-256)
    for sha, items in sha256_map.items():
        if len(items) > 1:
            report["exact_duplicates"].append({
                "sha256": sha,
                "paths": [it["path"] for it in items],
                "classes": list(set(it["class"] for it in items)),
            })

    # Detect near duplicates via imagehash (Hamming distance)
    if HAS_IMAGEHASH and len(phash_list) > 1:
        # Compare perceptual hashes
        for i in range(len(phash_list)):
            hash_i = imagehash.hex_to_hash(phash_list[i]["phash"])
            for j in range(i + 1, len(phash_list)):
                hash_j = imagehash.hex_to_hash(phash_list[j]["phash"])
                dist = hash_i - hash_j
                if dist <= hash_dist_threshold:
                    entry = {
                        "path1": phash_list[i]["path"],
                        "class1": phash_list[i]["class"],
                        "path2": phash_list[j]["path"],
                        "class2": phash_list[j]["class"],
                        "distance": int(dist),
                    }
                    report["near_duplicates"].append(entry)
                    if phash_list[i]["class"] != phash_list[j]["class"]:
                        report["cross_class_near_duplicates"].append(entry)

    report["summary"] = {
        "status": "PASS" if not report["corrupt_images"] and not report["tiny_images"] else "FAIL",
        "num_classes": len(report["classes"]),
        "corrupt_count": len(report["corrupt_images"]),
        "tiny_count": len(report["tiny_images"]),
        "exact_duplicate_groups": len(report["exact_duplicates"]),
        "near_duplicate_pairs": len(report["near_duplicates"]),
        "cross_class_near_duplicates": len(report["cross_class_near_duplicates"]),
    }
    return report


def validate_yolo(
    dataset_dir: Path, min_size: int, hash_dist_threshold: int
) -> Dict:
    """Validate a YOLO detection dataset."""
    report = {
        "layout": "yolo",
        "dataset_path": str(dataset_dir),
        "total_images": 0,
        "valid_images": 0,
        "missing_labels": [],
        "empty_labels": [],
        "invalid_bboxes": [],
        "corrupt_images": [],
        "tiny_images": [],
        "exact_duplicates": [],
        "classes": {},
        "summary": {},
    }

    # Find images
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    # Search for images either in images/ or across subdirectories
    img_candidates = []
    if images_dir.is_dir():
        img_candidates.extend([p for p in images_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS])
    else:
        img_candidates.extend([p for p in dataset_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS])

    sha256_map = defaultdict(list)
    class_counter = defaultdict(int)

    for img_p in img_candidates:
        report["total_images"] += 1
        rel_img = str(img_p.relative_to(dataset_dir))

        # Check image validity
        is_valid, err, dims, sha256, _ = validate_single_image(img_p, min_size=min_size)
        if not is_valid:
            if "Tiny image" in (err or ""):
                report["tiny_images"].append({"path": rel_img, "reason": err, "dimensions": dims})
            else:
                report["corrupt_images"].append({"path": rel_img, "reason": err})
            continue

        report["valid_images"] += 1
        sha256_map[sha256].append(rel_img)

        # Locate corresponding label file
        label_file = None
        if labels_dir.is_dir():
            # Match by relative subpath or stem
            cand = labels_dir / f"{img_p.stem}.txt"
            if cand.exists():
                label_file = cand
            else:
                # Check mirroring relative subpath
                try:
                    rel_to_img = img_p.relative_to(images_dir)
                    mirrored = labels_dir / rel_to_img.parent / f"{img_p.stem}.txt"
                    if mirrored.exists():
                        label_file = mirrored
                except Exception:
                    pass
        else:
            cand = img_p.parent / f"{img_p.stem}.txt"
            if cand.exists():
                label_file = cand

        if label_file is None or not label_file.exists():
            report["missing_labels"].append(rel_img)
            continue

        # Parse YOLO label
        try:
            content = label_file.read_text(encoding="utf-8").strip()
            if not content:
                report["empty_labels"].append(str(label_file.relative_to(dataset_dir)))
                continue

            for line_idx, line in enumerate(content.splitlines(), start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    report["invalid_bboxes"].append({
                        "label_path": str(label_file.relative_to(dataset_dir)),
                        "line": line_idx,
                        "reason": f"Expected at least 5 tokens (class x y w h), found {len(parts)}",
                    })
                    continue

                class_id = parts[0]
                class_counter[str(class_id)] += 1

                try:
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    report["invalid_bboxes"].append({
                        "label_path": str(label_file.relative_to(dataset_dir)),
                        "line": line_idx,
                        "reason": "Non-numeric coordinate values",
                    })
                    continue

                # Coordinates must be normalized in [0.0, 1.0] and positive dimensions
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    report["invalid_bboxes"].append({
                        "label_path": str(label_file.relative_to(dataset_dir)),
                        "line": line_idx,
                        "coords": [xc, yc, w, h],
                        "reason": "YOLO coordinates out of valid [0.0, 1.0] range or non-positive size",
                    })
        except Exception as e:
            report["corrupt_images"].append({"path": str(label_file.relative_to(dataset_dir)), "reason": f"Label read error: {str(e)}"})

    report["classes"] = dict(class_counter)

    for sha, paths in sha256_map.items():
        if len(paths) > 1:
            report["exact_duplicates"].append({"sha256": sha, "paths": paths})

    report["summary"] = {
        "status": "PASS" if not report["corrupt_images"] and not report["missing_labels"] and not report["invalid_bboxes"] else "FAIL",
        "corrupt_count": len(report["corrupt_images"]),
        "missing_label_count": len(report["missing_labels"]),
        "empty_label_count": len(report["empty_labels"]),
        "invalid_bbox_count": len(report["invalid_bboxes"]),
        "exact_duplicate_groups": len(report["exact_duplicates"]),
    }
    return report


def validate_voc(
    dataset_dir: Path, min_size: int, hash_dist_threshold: int
) -> Dict:
    """Validate a Pascal VOC XML detection dataset."""
    report = {
        "layout": "voc",
        "dataset_path": str(dataset_dir),
        "total_images": 0,
        "valid_images": 0,
        "missing_labels": [],
        "empty_labels": [],
        "invalid_bboxes": [],
        "corrupt_images": [],
        "tiny_images": [],
        "exact_duplicates": [],
        "classes": {},
        "summary": {},
    }

    annot_dir = dataset_dir / "Annotations"
    jpeg_dir = dataset_dir / "JPEGImages"

    if jpeg_dir.is_dir():
        img_paths = [p for p in jpeg_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    else:
        img_paths = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]

    sha256_map = defaultdict(list)
    class_counter = defaultdict(int)

    for img_p in img_paths:
        report["total_images"] += 1
        rel_img = str(img_p.relative_to(dataset_dir))

        is_valid, err, dims, sha256, _ = validate_single_image(img_p, min_size=min_size)
        if not is_valid:
            if "Tiny image" in (err or ""):
                report["tiny_images"].append({"path": rel_img, "reason": err, "dimensions": dims})
            else:
                report["corrupt_images"].append({"path": rel_img, "reason": err})
            continue

        report["valid_images"] += 1
        sha256_map[sha256].append(rel_img)
        img_w, img_h = dims

        # Locate XML annotation
        xml_file = None
        if annot_dir.is_dir():
            cand = annot_dir / f"{img_p.stem}.xml"
            if cand.exists():
                xml_file = cand
        else:
            cand = img_p.parent / f"{img_p.stem}.xml"
            if cand.exists():
                xml_file = cand

        if xml_file is None or not xml_file.exists():
            report["missing_labels"].append(rel_img)
            continue

        rel_xml = str(xml_file.relative_to(dataset_dir))

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            objects = root.findall("object")
            if not objects:
                report["empty_labels"].append(rel_xml)
                continue

            for obj_idx, obj in enumerate(objects):
                name_elem = obj.find("name")
                c_name = name_elem.text.strip() if name_elem is not None and name_elem.text else "unknown"
                class_counter[c_name] += 1

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    report["invalid_bboxes"].append({
                        "xml_path": rel_xml,
                        "object_idx": obj_idx,
                        "reason": "Missing <bndbox> element",
                    })
                    continue

                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                except Exception:
                    report["invalid_bboxes"].append({
                        "xml_path": rel_xml,
                        "object_idx": obj_idx,
                        "reason": "Failed to parse bndbox coordinate numbers",
                    })
                    continue

                # Check VOC coordinate logic
                if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin or xmax > img_w or ymax > img_h:
                    report["invalid_bboxes"].append({
                        "xml_path": rel_xml,
                        "object_idx": obj_idx,
                        "class": c_name,
                        "coords": [xmin, ymin, xmax, ymax],
                        "image_size": [img_w, img_h],
                        "reason": f"Bbox coordinates out of image bounds or inverted (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})",
                    })
        except Exception as e:
            report["corrupt_images"].append({"path": rel_xml, "reason": f"Corrupt XML: {str(e)}"})

    report["classes"] = dict(class_counter)

    for sha, paths in sha256_map.items():
        if len(paths) > 1:
            report["exact_duplicates"].append({"sha256": sha, "paths": paths})

    report["summary"] = {
        "status": "PASS" if not report["corrupt_images"] and not report["missing_labels"] and not report["invalid_bboxes"] else "FAIL",
        "corrupt_count": len(report["corrupt_images"]),
        "missing_label_count": len(report["missing_labels"]),
        "empty_label_count": len(report["empty_labels"]),
        "invalid_bbox_count": len(report["invalid_bboxes"]),
        "exact_duplicate_groups": len(report["exact_duplicates"]),
    }
    return report


def update_classes_yaml(crop_name: str, new_classes: List[str], task_type: str = "diseases"):
    """
    Safely update ml_training/configs/classes.yaml with real discovered classes only.
    Does NOT invent or fabricate any class names.
    """
    config_path = Path("ml_training/configs/classes.yaml")
    if not config_path.exists():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        crop_key = crop_name.lower().strip()
        if crop_key in data:
            current_list = set(data[crop_key].get(task_type, []))
            updated = sorted(list(current_list.union(set(new_classes))))
            data[crop_key][task_type] = updated

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False)
            print(f"[INFO] Updated {config_path} for crop '{crop_key}' -> {task_type}: {len(updated)} classes.")
    except Exception as e:
        print(f"[WARNING] Could not update classes.yaml: {str(e)}")


def validate_dataset(
    dataset_name: str,
    base_dir: Optional[Path] = None,
    min_size: int = 32,
    hash_dist: int = 2,
    output_dir: Optional[Path] = None,
    crop_name: Optional[str] = None,
) -> Dict:
    """
    Main validation entry point for a single dataset.
    """
    if base_dir is None:
        base_dir = Path("ml_training/datasets/raw")

    dataset_path = base_dir / dataset_name if not Path(dataset_name).is_absolute() and not Path(dataset_name).exists() else Path(dataset_name)

    # 1. Check if dataset exists and is not empty
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory not found: '{dataset_path}'")
        print(f"Please download real datasets and place them under 'ml_training/datasets/raw/{dataset_name}'.")
        print("Never use placeholder images.")
        return {"error": "Dataset folder missing", "path": str(dataset_path)}

    layout, meta = detect_layout(dataset_path)
    if layout in ["empty", "missing"]:
        print(f"[ERROR] Dataset directory '{dataset_path}' is EMPTY or missing.")
        print(f"Please extract the dataset files into '{dataset_path}' before validating.")
        return {"error": "Dataset folder empty", "path": str(dataset_path)}

    if layout == "unknown":
        print(f"[ERROR] Unrecognized dataset layout in '{dataset_path}'.")
        print("Expected one of: Classification (class_name/*.jpg), YOLO (images/ + labels/), or Pascal VOC (Annotations/ + JPEGImages/).")
        return {"error": "Unrecognized layout", "path": str(dataset_path)}

    print(f"\n==================================================")
    print(f" Validating Dataset: '{dataset_path.name}'")
    print(f" Detected Layout:    {layout.upper()}")
    print(f" Location:           {dataset_path.resolve()}")
    print(f"==================================================")

    if layout == "classification":
        report = validate_classification(dataset_path, min_size=min_size, hash_dist_threshold=hash_dist)
    elif layout == "yolo":
        report = validate_yolo(dataset_path, min_size=min_size, hash_dist_threshold=hash_dist)
    elif layout == "voc":
        report = validate_voc(dataset_path, min_size=min_size, hash_dist_threshold=hash_dist)
    else:
        report = {"error": f"Unsupported layout: {layout}"}

    # Save JSON report
    if output_dir is None:
        output_dir = Path("ml_training/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"{dataset_path.name}_validation.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Validation Report written to: {report_file}")
    print(f"Total Images:     {report.get('total_images', 0)}")
    print(f"Valid Images:     {report.get('valid_images', 0)}")
    print(f"Corrupt Images:   {len(report.get('corrupt_images', []))}")
    print(f"Tiny Images:      {len(report.get('tiny_images', []))}")
    print(f"Duplicate Groups: {len(report.get('exact_duplicates', []))}")
    if "near_duplicates" in report:
        print(f"Near-Duplicates:  {len(report['near_duplicates'])}")
    if "classes" in report:
        print(f"Classes Detected ({len(report['classes'])}): {list(report['classes'].keys())[:10]}")

    # Optional classes update
    if crop_name and "classes" in report and report["classes"]:
        update_classes_yaml(crop_name, list(report["classes"].keys()))

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate agricultural image datasets.")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset name inside ml_training/datasets/raw/ or path")
    parser.add_argument("--raw-dir", type=str, default="ml_training/datasets/raw", help="Base directory for raw datasets")
    parser.add_argument("--min-size", type=int, default=32, help="Minimum image dimension (px)")
    parser.add_argument("--hash-dist", type=int, default=2, help="Max perceptual hash distance for near-duplicates")
    parser.add_argument("--output-dir", type=str, default="ml_training/reports", help="Directory for JSON reports")
    parser.add_argument("--crop", type=str, default=None, help="Associated crop name (rice, wheat, maize, cotton, soybean, sugarcane)")
    parser.add_argument("--all", action="store_true", help="Validate all datasets found in raw-dir")

    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    if not raw_path.exists():
        print(f"[ERROR] Raw datasets directory does not exist: {raw_path}")
        print("Please create it and place real downloaded datasets inside.")
        sys.exit(1)

    if args.all:
        datasets = [d.name for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not datasets:
            print(f"[INFO] No datasets found in '{raw_path}'. Please place real datasets there.")
            sys.exit(0)
        for ds in datasets:
            validate_dataset(
                dataset_name=ds,
                base_dir=raw_path,
                min_size=args.min_size,
                hash_dist=args.hash_dist,
                output_dir=Path(args.output_dir),
                crop_name=args.crop,
            )
    elif args.dataset:
        report = validate_dataset(
            dataset_name=args.dataset,
            base_dir=raw_path,
            min_size=args.min_size,
            hash_dist=args.hash_dist,
            output_dir=Path(args.output_dir),
            crop_name=args.crop,
        )
        if "error" in report:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
