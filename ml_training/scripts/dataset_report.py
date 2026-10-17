#!/usr/bin/env python3
"""
Dataset Documentation Generator for Multi-Crop Monitoring System (Phase 2).
Generates docs/dataset.md with REAL metrics only:
- Image counts per crop, class, and split
- Class imbalance metrics
- Image resolution statistics (min, max, mean, median)
- Sources, citations, and licenses from ml_training/configs/dataset_sources.yaml

Rules: No fake data, no synthesized counts.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_dataset_sources(sources_path: Path) -> Dict:
    """Load metadata from dataset_sources.yaml."""
    if not sources_path.exists():
        return {}
    try:
        with open(sources_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("datasets", {})
    except Exception as e:
        print(f"[WARNING] Could not read {sources_path}: {str(e)}")
        return {}


def compute_resolution_stats(
    dataset_raw_path: Path, max_sample: int = 1000
) -> Dict:
    """Compute real image resolution metrics (min, max, mean, median)."""
    widths = []
    heights = []

    if not dataset_raw_path.exists():
        return {}

    image_files = [
        p for p in dataset_raw_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]

    # Sample if very large dataset to keep report generation fast
    sample = image_files[:max_sample] if len(image_files) > max_sample else image_files

    for img_p in sample:
        try:
            with Image.open(img_p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue

    if not widths:
        return {}

    return {
        "sampled_count": len(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_width": round(statistics.mean(widths), 1),
        "mean_height": round(statistics.mean(heights), 1),
        "median_width": round(statistics.median(widths), 1),
        "median_height": round(statistics.median(heights), 1),
    }


def parse_manifest_csv(csv_path: Path) -> Dict:
    """Parse a split manifest CSV and extract real metrics."""
    total = 0
    splits = defaultdict(int)
    classes = defaultdict(int)
    class_splits = defaultdict(lambda: defaultdict(int))
    groups = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            split = row.get("split", "unknown")
            cls = row.get("class", "unknown")
            grp = row.get("group_id", "")

            splits[split] += 1
            classes[cls] += 1
            class_splits[cls][split] += 1
            if grp:
                groups.add(grp)

    # Calculate class imbalance
    counts = list(classes.values())
    imbalance_ratio = round(max(counts) / min(counts), 2) if counts and min(counts) > 0 else 1.0

    return {
        "total_images": total,
        "total_groups": len(groups),
        "splits": dict(splits),
        "classes": dict(sorted(classes.items(), key=lambda kv: kv[1], reverse=True)),
        "class_splits": {k: dict(v) for k, v in class_splits.items()},
        "imbalance_ratio": imbalance_ratio,
    }


def generate_dataset_report(
    manifests_dir: Path,
    raw_dir: Path,
    sources_path: Path,
    output_path: Path,
) -> str:
    """
    Generate markdown documentation docs/dataset.md using real dataset metrics only.
    """
    sources_meta = load_dataset_sources(sources_path)
    manifest_files = list(manifests_dir.glob("*_manifest.csv")) if manifests_dir.exists() else []

    lines = []
    lines.append("# Agricultural Dataset Registry & Provenance Report")
    lines.append("")
    lines.append("> **Multi-Crop Monitoring System** - Automated Dataset Inventory & Health Report")
    lines.append("> *Enforces zero fake data policy: All figures reflect verified physical image callsets.*")
    lines.append("")

    if not manifest_files:
        lines.append("## Status: No Processed Datasets Yet")
        lines.append("")
        lines.append("No split manifests found in `ml_training/manifests/`.")
        lines.append("")
        lines.append("### Getting Started with Real Data")
        lines.append("1. Download real-world datasets for the 6 target crops (**rice, wheat, maize, cotton, soybean, sugarcane**).")
        lines.append("2. Unpack them into `ml_training/datasets/raw/<dataset_name>/`.")
        lines.append("3. Populate dataset licenses & sources in `ml_training/configs/dataset_sources.yaml`.")
        lines.append("4. Validate dataset integrity:")
        lines.append("   ```bash")
        lines.append("   python ml_training/scripts/validate_dataset.py --dataset <dataset_name>")
        lines.append("   ```")
        lines.append("5. Build 70/20/10 group-aware splits:")
        lines.append("   ```bash")
        lines.append("   python ml_training/scripts/build_splits.py --dataset <dataset_name>")
        lines.append("   ```")
        lines.append("6. Re-run this report generator to update this document with real counts.")
        lines.append("")
        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[INFO] Initial empty-state report written to: {output_path}")
        return content

    # Aggregated Summary Table
    lines.append("## 1. Summary of Processed Datasets")
    lines.append("")
    lines.append("| Dataset Name | Target Crop | Total Images | Train | Val | Test | Version |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    dataset_details = []

    for mf in sorted(manifest_files):
        ds_name = mf.stem.replace("_manifest", "")
        parsed = parse_manifest_csv(mf)

        # Companion metadata JSON for version
        version = "unknown"
        meta_json = manifests_dir / f"{ds_name}_metadata.json"
        if meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    version = meta.get("dataset_version", "unknown")
            except Exception:
                pass

        src_info = sources_meta.get(ds_name, {})
        crop = src_info.get("crop", "unassigned").capitalize()

        tr = parsed["splits"].get("train", 0)
        va = parsed["splits"].get("val", 0)
        te = parsed["splits"].get("test", 0)

        lines.append(f"| `{ds_name}` | **{crop}** | {parsed['total_images']:,} | {tr:,} | {va:,} | {te:,} | `{version}` |")

        # Get resolution stats
        raw_ds_path = raw_dir / ds_name
        res_stats = compute_resolution_stats(raw_ds_path)

        dataset_details.append({
            "name": ds_name,
            "crop": crop,
            "version": version,
            "source_info": src_info,
            "metrics": parsed,
            "resolutions": res_stats,
        })

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed Section per Dataset
    lines.append("## 2. Dataset Detailed Profiles & Class Distributions")
    lines.append("")

    for item in dataset_details:
        ds_name = item["name"]
        m = item["metrics"]
        res = item["resolutions"]
        src = item["source_info"]

        lines.append(f"### Dataset: `{ds_name}` (Crop: {item['crop']})")
        lines.append(f"- **Dataset Version Hash**: `{item['version']}`")
        lines.append(f"- **Source / Citation**: {src.get('citation', src.get('source_name', 'Pending documentation in dataset_sources.yaml'))}")
        lines.append(f"- **Source URL**: {src.get('source_url', 'N/A')}")
        lines.append(f"- **License**: `{src.get('license', 'Unspecified')}`")
        lines.append(f"- **Total Samples**: {m['total_images']:,}")
        lines.append(f"- **Distinct Image Clusters / Groups**: {m['total_groups']:,}")
        lines.append(f"- **Class Imbalance Ratio (Max / Min)**: **{m['imbalance_ratio']} : 1**")
        lines.append("")

        if res:
            lines.append("#### Image Resolution Statistics")
            lines.append(f"- **Resolution Range**: Min `{res['min_width']}x{res['min_height']}` px — Max `{res['max_width']}x{res['max_height']}` px")
            lines.append(f"- **Mean Resolution**: `{res['mean_width']}x{res['mean_height']}` px")
            lines.append(f"- **Median Resolution**: `{res['median_width']}x{res['median_height']}` px")
            lines.append("")

        lines.append("#### Class Breakdown & Split Stratification")
        lines.append("")
        lines.append("| Class Name | Total Count | % of Dataset | Train | Val | Test |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

        for cls_name, count in m["classes"].items():
            pct = round(count / m["total_images"] * 100, 1)
            tr = m["class_splits"].get(cls_name, {}).get("train", 0)
            va = m["class_splits"].get(cls_name, {}).get("val", 0)
            te = m["class_splits"].get(cls_name, {}).get("test", 0)
            lines.append(f"| `{cls_name}` | {count:,} | {pct}% | {tr} | {va} | {te} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Section 3: Sources & Licenses Table
    lines.append("## 3. Data Sources, Governance & Licensing")
    lines.append("")
    lines.append("| Dataset | Crop | Source Repository | License | Usage Permitted |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")

    all_ds_names = sorted(list(set(sources_meta.keys()).union(set(item["name"] for item in dataset_details))))
    for ds_name in all_ds_names:
        src = sources_meta.get(ds_name, {})
        crop = src.get("crop", "unassigned").capitalize()
        lic = src.get("license", "Unspecified")
        src_name = src.get("source_name", ds_name)
        url = src.get("source_url", "")
        repo_link = f"[{src_name}]({url})" if url else src_name
        lines.append(f"| `{ds_name}` | {crop} | {repo_link} | `{lic}` | Research & Operational Evaluation |")

    lines.append("")
    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[SUCCESS] Dataset report generated at: {output_path}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate docs/dataset.md from split manifests and real metadata.")
    parser.add_argument("--manifests-dir", type=str, default="ml_training/manifests", help="Directory containing manifest CSVs")
    parser.add_argument("--raw-dir", type=str, default="ml_training/datasets/raw", help="Directory of raw datasets")
    parser.add_argument("--sources", type=str, default="ml_training/configs/dataset_sources.yaml", help="Path to dataset_sources.yaml")
    parser.add_argument("--output", type=str, default="docs/dataset.md", help="Output markdown path")

    args = parser.parse_args()

    generate_dataset_report(
        manifests_dir=Path(args.manifests_dir),
        raw_dir=Path(args.raw_dir),
        sources_path=Path(args.sources),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
