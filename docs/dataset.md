# Agricultural Dataset Registry & Provenance Report

> **Multi-Crop Monitoring System** — Automated Dataset Inventory & Health Report
> *Enforces zero fake data policy: All figures reflect verified physical image callsets.*

---

## 1. Summary of Processed Datasets

| Dataset Name | Target Crop | Task Layout | Total Images | Train | Val | Test | Version Hash | Leakage Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rice_pest_lakthuru` | **Rice** | YOLO Detection (6 classes) | 5,229 | 3,660 | 1,046 | 523 | `f4b82c9e7102` | **0 leaks** (Verified Group-Aware) |
| `sugarcane_sppu` | **Sugarcane** | Classification (5 classes) | 2,521 | 1,762 | 502 | 257 | `e8310f7a9d41` | **0 leaks** (`use_for_eval` active) |
| `wheat_zenodo_jic` | **Wheat** | Classification (5 classes) | 999 | 697 | 197 | 105 | `d5dbead715ee` | **0 leaks** (Group-Aware Verified) |
| `cotton_mendeley` | **Cotton** | Classification (4 disease classes) | 980 | 685 | 196 | 99 | `a108f9c32b57` | **0 leaks** (Group-Aware Verified) |

---

## 2. Dataset Detailed Profiles & Class Distributions

### Dataset: `rice_pest_lakthuru` (Crop: Rice)
- **Dataset Version Hash**: `f4b82c9e7102` (Group-Aware Split)
- **Source / Citation**: Laktharu (2023). Rice Pests Dataset. Roboflow Universe.
- **Source URL**: https://universe.roboflow.com/laktharu/rice-pests-ztbeq
- **License**: `CC BY 4.0` (Verified 17-10-2026)
- **Total Images**: 5,229
- **Image Resolution**: $640 \times 640$ px (uniform Roboflow square format)

#### Data Leakage Remediation & Audit
- **Author's Original Split Leakage**:
  - Exact cross-split duplicate pairs: **185**
  - Near-duplicate cross-split pairs (Hamming distance $\le 2$): **731**
  - Cause: Augmented variations of the same capture were scattered across train and valid/test folders.
- **New Group-Aware Partition**:
  - All 5,229 images pooled and clustered via Union-Find using filename source IDs (e.g. `69541`, `bhopper-86`, `sborer-268`) and perceptual hash distance (Hamming $\le 4$).
  - Post-split cross-split exact duplicate pairs: **0**
  - Post-split cross-split near-duplicate pairs (Hamming $\le 4$): **0**
- **Degenerate Bounding Boxes**:
  - 8 bounding boxes in `train/labels/STEM-BORER_original_sborer-...` had $width = 0.0$.
  - Original label files remain strictly untouched (zero modification policy).
  - An exclusion registry [`ml_training/configs/rice_pest_excluded_boxes.json`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/configs/rice_pest_excluded_boxes.json) provides runtime filtering for YOLO loaders.
- **YOLO Training Inputs**:
  - File lists: [`train.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/train.txt), [`val.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/val.txt), [`test.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/test.txt)
  - Data config: [`data_groupsplit.yaml`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/data_groupsplit.yaml)

#### Pest Class Bounding Box Breakdown (Clean Boxes)

| Pest Class | Total Boxes | % of Annotations | Train Boxes | Val Boxes | Test Boxes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `whorl-maggot` | 1,616 | 20.0% | 1,131 | 323 | 162 |
| `brown-planthopper` | 1,578 | 19.5% | 1,105 | 316 | 157 |
| `leaf-folder` | 1,459 | 18.0% | 1,021 | 292 | 146 |
| `rice-bug` | 1,294 | 16.0% | 906 | 259 | 129 |
| `stem-borer` | 1,072 | 13.2% | 750 | 215 | 107 |
| `green-leafhopper` | 1,077 | 13.3% | 754 | 215 | 108 |
| **Total Boxes** | **8,096** | **100.0%** | **5,667** | **1,620** | **809** |

---

### Dataset: `sugarcane_sppu` (Crop: Sugarcane)
- **Dataset Version Hash**: `e8310f7a9d41`
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
- **Dataset Version Hash**: `a108f9c32b57`
- **Source / Citation**: Shingne, S., Thakur, P., Jangid, A., Buchade, P., Jain, A., & Agarwal, M. (2025). Cotton Leaf Disease Dataset with Severity Levels. Mendeley Data, V3. doi: 10.17632/74jsdxtmx2.3
- **Source URL**: https://doi.org/10.17632/74jsdxtmx2.3
- **License**: `CC BY 4.0` (Verified 17-10-2026)
- **Total Real Images**: 980 (Augmented ZIP archives excluded)
- **Class Structure**: 4 disease-only targets (`healthy`, `bacterial blight`, `fussarium wilt`, `curl virus`). Severity (`critical`, `mild`, `moderate`) is preserved as a metadata column `severity` in the manifest.
- **Display Name Mapping**: `"fussarium wilt"` $\to$ `"Fusarium wilt"` configured in [`ml_training/configs/classes.yaml`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/configs/classes.yaml).

#### Severe Class Imbalance Warning
> [!WARNING]
> Ratio between majority (`healthy`: 789) and minority (`curl virus`: 25) is **31.56 : 1**.
> In evaluation splits:
> - `curl virus`: **5** validation images, **3** test images ($< 10$ samples).
> - `fussarium wilt`: **9** validation images, **5** test images ($< 10$ samples).

#### Train-Split Class Weights ($N_{\text{train}} = 685$, $K = 4$)
Computed strictly on the `train` split using balanced inverse frequency:
$$w_c = \frac{N_{\text{train}}}{4 \times N_{c, \text{train}}}$$

| Disease Class | Total Images | Train | Val | Test | Train Split Weight $w_c$ | Severity Subfolders |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `healthy` | 789 | 552 | 158 | 79 | **0.3102** | *(none)* |
| `bacterial blight` | 119 | 83 | 24 | 12 | **2.0633** | critical: 19, mild: 50, moderate: 50 |
| `fussarium wilt` | 47 | 33 | 9 | 5 | **5.1894** | critical: 11, mild: 19, moderate: 17 |
| `curl virus` | 25 | 17 | 5 | 3 | **10.0735** | critical: 11, mild: 9, moderate: 5 |
| **Total** | **980** | **685** | **196** | **99** | — | — |

---

## 3. Data Governance & Attribution

| Dataset | Target Crop | Source Reference | License | Integrity Status |
| :--- | :--- | :--- | :--- | :--- |
| `rice_pest_lakthuru` | Rice | [Roboflow Universe (Laktharu)](https://universe.roboflow.com/laktharu/rice-pests-ztbeq) | `CC BY 4.0` | Group-aware split; 0 leakage; 8 degenerate bboxes excluded |
| `sugarcane_sppu` | Sugarcane | [Mendeley Data V1 (SPPU)](https://doi.org/10.17632/9424skmnrk.1) | `CC BY 4.0` | 2,521 physical images reconciled; `use_for_eval` active |
| `wheat_zenodo_jic` | Wheat | [Zenodo (John Innes Centre)](https://doi.org/10.5281/zenodo.7573133) | `CC BY 4.0` | Verified 999 images; leak-free split |
| `cotton_mendeley` | Cotton | [Mendeley Data V3 (Shingne et al.)](https://doi.org/10.17632/74jsdxtmx2.3) | `CC BY 4.0` | 4 disease classes; severity in metadata; train class weights |
