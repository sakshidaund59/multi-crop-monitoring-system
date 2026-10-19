# Agricultural Dataset Registry & Provenance Report

> **Multi-Crop Monitoring System** — Automated Dataset Inventory & Health Report
> *Enforces zero fake data policy: All figures reflect verified physical image callsets.*

---

## 1. Summary of Processed Datasets

| Dataset Name | Target Crop | Task Layout | Total Images | Train | Val | Test | Version Hash | Leakage Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rice_pest_lakthuru` | **Rice** | YOLO Detection (6 classes) | 5,229 | 3,639 | 1,033 | 557 | `6b9e5c4ed209` | Hardened v2 (No leaks at pHash $\le 4$ or Cosine $\ge 0.95$) |
| `sugarcane_sppu` | **Sugarcane** | Classification (5 classes) | 2,521 | 1,762 | 502 | 257 | `c2396e678992` | Group-Aware Verified (`use_for_eval` active) |
| `wheat_zenodo_jic` | **Wheat** | Classification (5 classes) | 999 | 697 | 197 | 105 | `d5dbead715ee` | Group-Aware Verified |
| `cotton_mendeley` | **Cotton** | Classification (4 disease classes) | 980 | 686 | 196 | 98 | `99c7fa25ceeb` | Group-Aware Verified (Deterministic split) |

---

## 2. Dataset Detailed Profiles & Class Distributions

### Dataset: `rice_pest_lakthuru` (Crop: Rice)
- **Dataset Version Hash**: `6b9e5c4ed209` (Hardened Group-Aware Split v2)
- **Source / Citation**: Laktharu (2023). Rice Pests Dataset. Roboflow Universe.
- **Source URL**: https://universe.roboflow.com/laktharu/rice-pests-ztbeq
- **License**: `CC BY 4.0` (Verified 17-10-2026)
- **Total Images**: 5,229 (Train: 3,639 | Val: 1,033 | Test: 557)
- **Image Resolution**: $640 \times 640$ px (uniform Roboflow square format)
- **Tabular Leakage Audit**: Complete metrics report in [`docs/leakage_audit_summary.md`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/docs/leakage_audit_summary.md)

#### Data Leakage Remediation & Audit
- **Author's Original Split Leakage**:
  - Exact cross-split duplicate pairs: **185**
  - Near-duplicate cross-split pairs (Hamming distance $\le 2$): **731**
  - Cause: Augmented variations of the same capture were scattered across train and valid/test folders.
- **New Group-Aware Partition**:
  - All 5,229 images pooled and clustered via Union-Find using canonical filename source IDs (e.g. `69541`, `bhopper-86`, `sborer-268`, with `-png`/`-jpeg` stripped) and perceptual hash distance (Hamming $\le 4$).
  - Post-split cross-split exact duplicate pairs: **0**
  - Post-split cross-split near-duplicate pairs (Hamming $\le 4$): **0**
  - Post-split cross-split CNN nearest-neighbor pairs (Cosine $\ge 0.95$): **0**
  - Boundary claim: *No cross-split near-duplicates at perceptual-hash distance $\le 4$ or CNN cosine $\ge 0.95$ (residual similarity remains at looser thresholds: hash $T \le 6 = 86$ pairs, cosine $\ge 0.90 = 86$ val / 22 test images).*
- **Degenerate Bounding Boxes**:
  - 8 bounding boxes in `train/labels/STEM-BORER_original_sborer-...` had $width = 0.0$.
  - Original label files remain strictly untouched (zero modification policy).
  - An exclusion registry [`ml_training/configs/rice_pest_excluded_boxes.json`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/configs/rice_pest_excluded_boxes.json) provides runtime filtering for YOLO loaders.
- **YOLO Training Inputs**:
  - File lists: [`train.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/train.txt), [`val.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/val.txt), [`test.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/test.txt)
  - Data config: [`data_groupsplit_v2.yaml`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/data_groupsplit_v2.yaml)

#### Pest Class Bounding Box Breakdown (Clean Boxes, v2 Partition)

| Pest Class | Total Clean Boxes | % of Annotations | Train Boxes | Val Boxes | Test Boxes | Val/Test Health Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `whorl-maggot` | 1,616 | 20.0% | 1,139 | 330 | 147 | **WARNING**: < 15 clusters in val (13) & test (3) — Indicative only |
| `brown-planthopper` | 1,578 | 19.5% | 1,086 | 357 | 135 | **WARNING**: < 15 clusters in val (7) & test (13) — Indicative only |
| `leaf-folder` | 1,459 | 18.0% | 1,031 | 300 | 128 | Passed ($\ge 100$ boxes, $\ge 15$ clusters) |
| `rice-bug` | 1,294 | 16.0% | 916 | 266 | 112 | Passed ($\ge 100$ boxes, $\ge 15$ clusters) |
| `stem-borer` | 1,072 | 13.2% | 758 | 220 | 94 | **WARNING**: < 100 boxes (94) & < 15 clusters (12) in test — Indicative only |
| `green-leafhopper` | 1,077 | 13.3% | 762 | 220 | 95 | **WARNING**: < 100 boxes in test (95 boxes) — Indicative only |
| **Total Boxes** | **8,096** | **100.0%** | **5,692** | **1,693** | **711** | **Evaluation warnings active for 4 classes** |

> [!NOTE]
> Metrics for classes with active warnings (`brown-planthopper`, `green-leafhopper`, `stem-borer`, `whorl-maggot`) are indicative only due to bounding box counts < 100 or independent cluster counts < 15 in those specific evaluation partitions. Two classes (`leaf-folder`, `rice-bug`) satisfy both $\ge 100$ boxes and $\ge 15$ clusters across all evaluation splits.

---

### Dataset: `sugarcane_sppu` (Crop: Sugarcane)
- **Dataset Version Hash**: `c2396e678992`
- **Source / Citation**: Daphal, S., & Koli, S. (2022). Sugarcane Leaf Disease Dataset. Mendeley Data, V1. doi: 10.17632/9424skmnrk.1
- **Source URL**: https://doi.org/10.17632/9424skmnrk.1
- **License**: `CC BY 4.0` (Verified 17-10-2026)
- **Total Samples**: 2,521
- **Class Imbalance Ratio (Max / Min)**: **1.13 : 1**

#### Reconciliation (2,521 Physical Images vs 2,569 Mendeley Description)
The official Mendeley Data web description lists 2,569 images ($524 + 470 + 524 + 530 + 521$). Physical unpacking reveals 2,521 valid images. Exactly **48 sequential numbering gaps** exist where the original authors purged corrupt or unreadable captures before upload without re-indexing:
- `Healthy`: 522 images (missing indices: 440, 476)
- `Mosaic`: 462 images (missing indices: 18, 19, 40, 122, 155, 293, 348, 418)
- `RedRot`: 518 images (missing indices: 17, 105, 199, 231, 370, 488)
- `Rust`: 514 images (missing indices: 3, 27, 74, 96, 133, 170, 204, 242, 280, 319, 356, 394, 433, 469, 505, 519)
- `Yellow`: 505 images (missing indices: 8, 39, 76, 111, 149, 185, 223, 260, 298, 335, 372, 410, 448, 484, 501, 512)
*All 2,521 images on disk are 100% verified, genuine plant photos.*

#### Duplicate Evaluation Policy (`use_for_eval`)
All duplicate image groups sit strictly within the same split (0 cross-split leakage). To prevent repeated images from inflating validation or test metrics:
- Manifest column `use_for_eval` is `true` for all train images.
- In `val` and `test`, exactly ONE image per exact-duplicate group is flagged `use_for_eval = true`. Duplicate copies are flagged `use_for_eval = false`.

#### Class Breakdown & Split Stratification

| Class Name | Total Count | % of Dataset | Train | Val | Test | Val/Test Eval Reps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Healthy` | 522 | 20.7% | 365 | 104 | 53 | 147 |
| `RedRot` | 518 | 20.5% | 362 | 103 | 53 | 156 |
| `Rust` | 514 | 20.4% | 359 | 102 | 53 | 155 |
| `Yellow` | 505 | 20.0% | 353 | 101 | 51 | 152 |
| `Mosaic` | 462 | 18.3% | 323 | 92 | 47 | 139 |

---

### Dataset: `wheat_zenodo_jic` (Crop: Wheat)
- **Dataset Version Hash**: `d5dbead715ee`
- **Source / Citation**: Long, M. C., Hartley, M., Morris, R. J., & Brown, J. K. M. (2023). Wheat disease images (small dataset) - Long 2023 Plant Path 999 photos. Zenodo. doi: 10.5281/zenodo.7573133
- **Source URL**: https://doi.org/10.5281/zenodo.7573133
- **License**: `CC BY 4.0` (Verified 17-10-2026)
- **Total Samples**: 999
- **Distinct Image Clusters / Groups**: 999
- **Class Imbalance Ratio (Max / Min)**: **2.86 : 1**

#### Class Breakdown & Split Stratification

| Class Name | Total Count | % of Dataset | Train | Val | Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Septoria` | 349 | 34.9% | 244 | 69 | 36 |
| `YellowRust` | 239 | 23.9% | 167 | 47 | 25 |
| `Mildew` | 161 | 16.1% | 112 | 32 | 17 |
| `BrownRust` | 128 | 12.8% | 89 | 25 | 14 |
| `Healthy` | 122 | 12.2% | 85 | 24 | 13 |

---

### Dataset: `cotton_mendeley` (Crop: Cotton)
- **Dataset Version Hash**: `99c7fa25ceeb`
- **Source / Citation**: Shingne, S., Thakur, P., Jangid, A., Buchade, P., Jain, A., & Agarwal, M. (2025). Cotton Leaf Disease Dataset with Severity Levels. Mendeley Data, V3. doi: 10.17632/74jsdxtmx2.3
- **Source URL**: https://doi.org/10.17632/74jsdxtmx2.3
- **License**: `CC BY 4.0` (Verified 17-10-2026)
- **Total Real Images**: 980 (Augmented ZIP archives excluded)
- **Split Distribution**: Train: 686 | Val: 196 | Test: 98
- **Class Structure**: 4 disease-only targets (`healthy`, `bacterial blight`, `fussarium wilt`, `curl virus`). Severity (`critical`, `mild`, `moderate`) is preserved as a metadata column `severity` in the manifest.
- **Display Name Mapping**: `"fussarium wilt"` $\to$ `"Fusarium wilt"` configured in [`ml_training/configs/classes.yaml`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/configs/classes.yaml).

#### Severe Class Imbalance Warning
> [!WARNING]
> Ratio between majority (`healthy`: 789) and minority (`curl virus`: 25) is **31.56 : 1**.
> In evaluation splits:
> - `curl virus`: **5** validation images, **2** test images ($< 10$ samples).
> - `fussarium wilt`: **9** validation images, **5** test images ($< 10$ samples).

#### Train-Split Class Weights ($N_{\text{train}} = 686$, $K = 4$)
Computed strictly on the `train` split using balanced inverse frequency:
$$w_c = \frac{N_{\text{train}}}{4 \times N_{c, \text{train}}}$$

| Disease Class | Total Images | Train | Val | Test | Train Split Weight $w_c$ | Severity Subfolders |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `healthy` | 789 | 552 | 158 | 79 | **0.3107** | *(none)* |
| `bacterial blight` | 119 | 83 | 24 | 12 | **2.0663** | critical: 19, mild: 50, moderate: 50 |
| `fussarium wilt` | 47 | 33 | 9 | 5 | **5.1970** | critical: 11, mild: 19, moderate: 17 |
| `curl virus` | 25 | 18 | 5 | 2 | **9.5278** | critical: 11, mild: 9, moderate: 5 |
| **Total** | **980** | **686** | **196** | **98** | — | — |

---

## 3. Data Governance & Attribution

| Dataset | Target Crop | Source Reference | License | Integrity Status |
| :--- | :--- | :--- | :--- | :--- |
| `rice_pest_lakthuru` | Rice | [Roboflow Universe (Laktharu)](https://universe.roboflow.com/laktharu/rice-pests-ztbeq) | `CC BY 4.0` | Group-aware split v2; no leaks at pHash $\le 4$ or Cosine $\ge 0.95$; 8 degenerate bboxes excluded |
| `sugarcane_sppu` | Sugarcane | [Mendeley Data V1 (SPPU)](https://doi.org/10.17632/9424skmnrk.1) | `CC BY 4.0` | 2,521 physical images reconciled; `use_for_eval` active |
| `wheat_zenodo_jic` | Wheat | [Zenodo (John Innes Centre)](https://doi.org/10.5281/zenodo.7573133) | `CC BY 4.0` | Verified 999 images; group-aware partition |
| `cotton_mendeley` | Cotton | [Mendeley Data V3 (Shingne et al.)](https://doi.org/10.17632/74jsdxtmx2.3) | `CC BY 4.0` | 4 disease classes; severity in metadata; train class weights |
