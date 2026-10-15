#!/usr/bin/env python3
"""
Group-Aware Dataset Splitter for Multi-Crop Monitoring System (Phase 2).
Performs a 70/20/10 train/val/test split grouped by plant/source/perceptual-hash cluster
so that near-duplicates and burst-shots never cross split boundaries.
Stratifies across classes where feasible and generates a versioned manifest CSV.

Rules: No fake data, no synthesized labels or images.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_filename_prefix(filename: str) -> str:
    """
    Extract plant or burst shot prefix from filename.
    Examples:
        'plant12_leaf01.jpg' -> 'plant12'
        'DJI_0045_crop.png'  -> 'DJI_0045'
        'rice_blast_001.jpg' -> 'rice_blast_001'
    """
    stem = Path(filename).stem
    # Match common prefix patterns (e.g., prefix before last underscore with digits or plant IDs)
    match = re.match(r"^([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*?)(?:[_\-]\d+)?$", stem)
    if match:
        return match.group(1).lower()
    parts = stem.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1]).lower()
    return stem.lower()


class UnionFind:
    """Union-Find data structure for clustering connected near-duplicates."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def cluster_into_groups(
    images_info: List[Dict], hash_threshold: int = 3
) -> List[Dict]:
    """
    Assign group IDs using connected components of:
    1. Perceptual hash distance <= hash_threshold (near-duplicates)
    2. Filename prefix matching (same plant / shot sequence)
    """
    uf = UnionFind()
    n = len(images_info)

    # 1. Union by filename prefix
    prefix_to_indices = defaultdict(list)
    for idx, item in enumerate(images_info):
        prefix = extract_filename_prefix(item["path"])
        prefix_to_indices[prefix].append(idx)

    for prefix, indices in prefix_to_indices.items():
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # 2. Union by perceptual hash if available
    if HAS_IMAGEHASH and n > 1:
        hashes = []
        for item in images_info:
            ph = item.get("phash")
            if ph:
                try:
                    hashes.append(imagehash.hex_to_hash(ph))
                except Exception:
                    hashes.append(None)
            else:
                hashes.append(None)

        for i in range(n):
            if hashes[i] is None:
                continue
            for j in range(i + 1, n):
                if hashes[j] is None:
                    continue
                # If images belong to same class and have low Hamming distance
                if images_info[i]["class"] == images_info[j]["class"]:
                    dist = hashes[i] - hashes[j]
                    if dist <= hash_threshold:
                        uf.union(i, j)

    # 3. Form final group_id
    group_map = {}
    for idx in range(n):
        root = uf.find(idx)
        if root not in group_map:
            group_map[root] = f"grp_{len(group_map):05d}"
        images_info[idx]["group_id"] = group_map[root]

    return images_info


def stratify_groups_split(
    grouped_items: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
) -> List[Dict]:
    """
    Greedy group-aware stratified partition into train/val/test.
    Ensures that every item in a given group_id is strictly assigned to the SAME split.
    """
    # Group items by class, then by group_id
    # { class_name: { group_id: [item1, item2, ...] } }
    class_groups = defaultdict(lambda: defaultdict(list))
    for it in grouped_items:
        class_groups[it["class"]][it["group_id"]].append(it)

    for c_name, grps in class_groups.items():
        total_class_images = sum(len(items) for items in grps.values())
        if total_class_images < 10:
            print(
                f"[WARNING] Class '{c_name}' has only {total_class_images} images. "
                f"Minimum 10 images recommended for reliable 70/20/10 split."
            )

        target_train = int(total_class_images * train_ratio)
        target_val = int(total_class_images * val_ratio)
        # Remaining goes to test to ensure total == total_class_images
        target_test = total_class_images - (target_train + target_val)

        # Sort groups by size descending for smoother greedy bin packing
        sorted_groups = sorted(grps.items(), key=lambda kv: len(kv[1]), reverse=True)

        train_count = 0
        val_count = 0
        test_count = 0

        for grp_id, items in sorted_groups:
            k = len(items)
            # Choose bin that needs images most relative to targets
            # Priority: train if below target, then val, then test
            need_train = target_train - train_count
            need_val = target_val - val_count
            need_test = target_test - test_count

            # Assign to largest deficit
            deficits = [
                (need_train, "train", train_count),
                (need_val, "val", val_count),
                (need_test, "test", test_count),
            ]
            deficits.sort(key=lambda x: x[0], reverse=True)

            assigned_split = deficits[0][1]

            if assigned_split == "train":
                train_count += k
            elif assigned_split == "val":
                val_count += k
            else:
                test_count += k

            for it in items:
                it["split"] = assigned_split

    return grouped_items


def build_splits(
    dataset_name: str,
    raw_dir: Path,
    manifests_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    hash_threshold: int = 3,
) -> Dict:
    """
    Main function to construct group-aware splits.
    """
    dataset_path = raw_dir / dataset_name if not Path(dataset_name).is_absolute() else Path(dataset_name)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory does not exist: '{dataset_path}'")
        print("Please place real dataset files there before splitting.")
        return {"error": "Dataset missing"}

    print(f"\n==================================================")
    print(f" Building Splits for: '{dataset_path.name}'")
    print(f" Ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print(f"==================================================")

    # 1. Collect images and classes
    # Check if classification or detection
    images_info = []

    # Detect classification subdirectories
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if subdirs and not (dataset_path / "images").exists() and not (dataset_path / "Annotations").exists():
        # Classification folders
        for c_dir in sorted(subdirs, key=lambda d: d.name):
            c_name = c_dir.name
            img_files = [p for p in c_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
            for img_p in img_files:
                sha = compute_sha256(img_p)
                ph = None
                if HAS_IMAGEHASH:
                    try:
                        with Image.open(img_p) as im:
                            ph = str(imagehash.phash(im))
                    except Exception:
                        ph = None
                images_info.append({
                    "path": str(img_p.relative_to(dataset_path)).replace("\\", "/"),
                    "class": c_name,
                    "sha256": sha,
                    "phash": ph,
                })
    else:
        # Detection or flat layout: find all images and assign class if detectable
        all_imgs = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        for img_p in all_imgs:
            sha = compute_sha256(img_p)
            ph = None
            if HAS_IMAGEHASH:
                try:
                    with Image.open(img_p) as im:
                        ph = str(imagehash.phash(im))
                except Exception:
                    ph = None
            images_info.append({
                "path": str(img_p.relative_to(dataset_path)).replace("\\", "/"),
                "class": "object_detection",
                "sha256": sha,
                "phash": ph,
            })

    if not images_info:
        print(f"[ERROR] No valid images found in '{dataset_path}'.")
        return {"error": "No images found"}

    print(f"[INFO] Discovered {len(images_info)} images across classes.")

    # 2. Cluster into groups using perceptual hash + prefix
    clustered_items = cluster_into_groups(images_info, hash_threshold=hash_threshold)

    # 3. Stratify and split groups
    split_items = stratify_groups_split(
        clustered_items,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # 4. Sort rows deterministically by path
    split_items.sort(key=lambda x: x["path"])

    # 5. Build manifest CSV content
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifests_dir / f"{dataset_path.name}_manifest.csv"

    # Write CSV
    fieldnames = ["path", "class", "split", "sha256", "group_id"]
    csv_rows = []
    hasher = hashlib.sha256()

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in split_items:
            row = {
                "path": it["path"],
                "class": it["class"],
                "split": it["split"],
                "sha256": it["sha256"],
                "group_id": it["group_id"],
            }
            writer.writerow(row)
            # Hash line for dataset version
            line_str = f"{row['path']},{row['class']},{row['split']},{row['sha256']},{row['group_id']}\n"
            hasher.update(line_str.encode("utf-8"))

    # Compute short dataset_version hash
    dataset_version = hasher.hexdigest()[:12]

    # Calculate statistics
    stats = {
        "dataset_name": dataset_path.name,
        "dataset_version": dataset_version,
        "total_images": len(split_items),
        "total_groups": len(set(it["group_id"] for it in split_items)),
        "splits": defaultdict(int),
        "classes_per_split": defaultdict(lambda: defaultdict(int)),
    }

    for it in split_items:
        stats["splits"][it["split"]] += 1
        stats["classes_per_split"][it["class"]][it["split"]] += 1

    stats["splits"] = dict(stats["splits"])
    stats["classes_per_split"] = {k: dict(v) for k, v in stats["classes_per_split"].items()}

    # Save companion metadata JSON
    meta_json = manifests_dir / f"{dataset_path.name}_metadata.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[SUCCESS] Split Manifest saved to: {manifest_csv}")
    print(f"[SUCCESS] Dataset Version:        {dataset_version}")
    print(f"Total Images:    {stats['total_images']}")
    print(f"Total Groups:    {stats['total_groups']}")
    print(f"Train Images:    {stats['splits'].get('train', 0)} ({stats['splits'].get('train', 0)/len(split_items)*100:.1f}%)")
    print(f"Val Images:      {stats['splits'].get('val', 0)} ({stats['splits'].get('val', 0)/len(split_items)*100:.1f}%)")
    print(f"Test Images:     {stats['splits'].get('test', 0)} ({stats['splits'].get('test', 0)/len(split_items)*100:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build group-aware 70/20/10 train/val/test splits.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name inside raw/ or path")
    parser.add_argument("--raw-dir", type=str, default="ml_training/datasets/raw", help="Directory of raw datasets")
    parser.add_argument("--manifests-dir", type=str, default="ml_training/manifests", help="Output directory for manifests")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Val split ratio (default: 0.20)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test split ratio (default: 0.10)")
    parser.add_argument("--hash-threshold", type=int, default=3, help="Max perceptual hash distance for clustering")

    args = parser.parse_args()

    # Validate ratios
    if not abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) < 1e-5:
        print("[ERROR] Split ratios must sum to 1.0 (e.g. 0.70 0.20 0.10)")
        sys.exit(1)

    result = build_splits(
        dataset_name=args.dataset,
        raw_dir=Path(args.raw_dir),
        manifests_dir=Path(args.manifests_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hash_threshold=args.hash_threshold,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
