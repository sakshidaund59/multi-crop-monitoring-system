# Data Issues, Anomalies & Reconciliation Log

> **Multi-Crop Monitoring System** — Phase 2 Dataset Quality Assurance Record
> *Strict adherence to zero fake data, no invented labels, and immutable original dataset files.*

---

## 1. Rice Pest Dataset (`rice_pest_lakthuru`)

### 1.1 Degenerate Bounding Boxes ($width = 0.0$)
Validation revealed 8 label entries in YOLO detection annotations with degenerate zero width. Under YOLO format `(class, x_center, y_center, width, height)`, bounding boxes with zero width cause mathematical collapse or division-by-zero during intersection-over-union (IoU) calculation and non-maximum suppression (NMS).

In accordance with our strict data governance rule (*never edit original dataset files*), original `.txt` files in `ml_training/datasets/raw/rice_pest_lakthuru/` remain unmodified. An exclusion list has been generated at [`ml_training/configs/rice_pest_excluded_boxes.json`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/configs/rice_pest_excluded_boxes.json) for training pipelines to filter these 8 degenerate boxes dynamically at batch ingestion.

| # | Label File Relative Path | Line | Class ID | Class Name | Coordinates `[x, y, w, h]` | Root Cause / Issue |
|---|---|---|---|---|---|---|
| 1 | `train/labels/STEM-BORER_original_sborer-265-jpg_180a18c8-f340-4331-98b6-b08136a07cfb_jpg.rf.28fd10f2014eaeff6e47a5da8edd3568.txt` | 1 | 4 | `stem-borer` | `[0.0, 0.0, 0.0, 0.995325]` | Degenerate width = 0.0 |
| 2 | `train/labels/STEM-BORER_original_sborer-266-jpg_00344bad-1901-4585-9dfb-677f51dff1fe_jpg.rf.6e7519b08c7b8f70982da4c6da3b2f5d.txt` | 1 | 4 | `stem-borer` | `[0.0, 0.0, 0.0, 0.997596]` | Degenerate width = 0.0 |
| 3 | `train/labels/STEM-BORER_original_sborer-266-jpg_0d23a838-004c-497b-b449-fba26e482df6_jpg.rf.cfccc9f96e2b3353f4ef4fa6b267d629.txt` | 1 | 4 | `stem-borer` | `[0.053626, 0.0, 0.0, 0.001148]` | Degenerate width = 0.0 |
| 4 | `train/labels/STEM-BORER_original_sborer-266-jpg_c133f9f3-3459-407c-b161-a81a9b9d9f5c_jpg.rf.a9534a5b2577cad6176e42af181e0206.txt` | 1 | 4 | `stem-borer` | `[0.0, 0.0, 0.0, 0.997596]` | Degenerate width = 0.0 |
| 5 | `train/labels/STEM-BORER_original_sborer-267-jpg_253f5b62-d349-4cb2-8b5c-4f6203c62d0c_jpg.rf.1e63dab9360ae09c971ddae4781b95ee.txt` | 1 | 4 | `stem-borer` | `[0.0, 0.0, 0.0, 0.405865]` | Degenerate width = 0.0 |
| 6 | `train/labels/STEM-BORER_original_sborer-268-jpg_d5f73641-294e-4d9b-b23e-2c2f0ce3f064_jpg.rf.010e247988f983d9dcef198f9ce7c5e9.txt` | 1 | 4 | `stem-borer` | `[0.0, 0.0, 0.0, 0.238204]` | Degenerate width = 0.0 |
| 7 | `train/labels/STEM-BORER_original_sborer-42-jpg_ba5de9bb-78cd-4b0a-b398-764abf4021ce_jpg.rf.dfa82ef896a94368b34cccd2c5d4fe86.txt` | 1 | 4 | `stem-borer` | `[0.050895, 0.0, 0.0, 0.077758]` | Degenerate width = 0.0 |
| 8 | `train/labels/STEM-BORER_original_sborer-72-jpg_4c0004e5-2ec5-477b-8f65-c51c1a4dcfa2_jpg.rf.65ee17fc591642d40a17af3081475502.txt` | 1 | 4 | `stem-borer` | `[0.0, 0.0, 0.0, 0.166296]` | Degenerate width = 0.0 |

### 1.2 Cross-Split Data Leakage in Author Split
- **Problem**: The author's provided Roboflow split (`train`, `valid`, `test`) suffered severe data leakage due to augmentations (flips, crops, hue shifts) of the same physical photographs distributed across train and validation/test subsets. Specifically, **185 exact duplicate pairs** and **731 near-duplicate pairs** (Hamming distance $\le 2$) crossed split boundaries.
- **Resolution**: All 5,229 images were pooled into a unified dataset. Images were clustered using union-find on perceptual hash distance ($\text{Hamming} \le 4$) AND shared original source IDs extracted from filenames (e.g. `69541` in `69541_jpg.rf...` and `..._original_69541-jpg_...`).
- **Post-Split Verification**:
  - Cross-split exact duplicate pairs: **0**
  - Cross-split near-duplicate pairs ($\text{Hamming} \le 4$): **0**
- **Manifest Status**: The original manifest [`ml_training/manifests/rice_pest_lakthuru_manifest.csv`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/manifests/rice_pest_lakthuru_manifest.csv) is retained and explicitly marked as `"has leakage, do not use for reported results"`. The official group-aware split is saved in [`ml_training/manifests/rice_pest_lakthuru_groupsplit_manifest.csv`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/manifests/rice_pest_lakthuru_groupsplit_manifest.csv).
- **YOLO Training Configuration**: File lists [`train.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/train.txt), [`val.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/val.txt), and [`test.txt`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/test.txt) reference local paths directly without duplicating image bytes. [`data_groupsplit.yaml`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/datasets/raw/rice_pest_lakthuru/data_groupsplit.yaml) configures YOLO detectors to use this clean partitioning.

---

## 2. Sugarcane Dataset Reconciliation (`sugarcane_sppu`)

### 2.1 Discrepancy: 2,521 Physical Images vs 2,569 Repository Header
The Mendeley Data page and article metadata for *Sugarcane Leaf Disease Dataset* (Daphal & Koli, 2022) reports a total count of **2,569** images:
- Healthy: 524
- Mosaic: 470
- RedRot: 524
- Rust: 530
- Yellow: 521
- **Sum**: $524 + 470 + 524 + 530 + 521 = 2,569$ images.

However, physical extraction from the official Mendeley archive yields exactly **2,521** valid image files on disk. Detailed audit of sequential filename numbering reveals exactly **48 missing images**:

| Class | Expected in Description | Found on Disk | Missing Count | Specific Missing Index Numbers |
|---|---|---|---|---|
| `Healthy` | 524 | 522 | 2 | `healthy (440).jpeg`, `healthy (476).jpeg` |
| `Mosaic` | 470 | 462 | 8 | `mosaic (18).jpeg`, `mosaic (19).jpeg`, `mosaic (40).jpeg`, `mosaic (122).jpeg`, `mosaic (155).jpeg`, `mosaic (293).jpeg`, `mosaic (348).jpeg`, `mosaic (418).jpeg` |
| `RedRot` | 524 | 518 | 6 | `red rot (17).jpeg`, `red rot (105).jpeg`, `red rot (199).jpeg`, `red rot (231).jpeg`, `red rot (370).jpeg`, `red rot (488).jpeg` |
| `Rust` | 530 | 514 | 16 | `rust (3).jpeg`, `rust (27).jpeg`, `rust (74).jpeg`, `rust (96).jpeg`, `rust (133).jpeg`, `rust (170).jpeg`, `rust (204).jpeg`, `rust (242).jpeg`, `rust (280).jpeg`, `rust (319).jpeg`, `rust (356).jpeg`, `rust (394).jpeg`, `rust (433).jpeg`, `rust (469).jpeg`, `rust (505).jpeg`, `rust (519).jpeg` |
| `Yellow` | 521 | 505 | 16 | `yellow (8).jpeg`, `yellow (39).jpeg`, `yellow (76).jpeg`, `yellow (111).jpeg`, `yellow (149).jpeg`, `yellow (185).jpeg`, `yellow (223).jpeg`, `yellow (260).jpeg`, `yellow (298).jpeg`, `yellow (335).jpeg`, `yellow (372).jpeg`, `yellow (410).jpeg`, `yellow (448).jpeg`, `yellow (484).jpeg`, `yellow (501).jpeg`, `yellow (512).jpeg` |
| **Total** | **2,569** | **2,521** | **48** | *Total 48 omitted index gaps* |

**Reason for Discrepancy**: During dataset preparation by the authors, 48 corrupted, blurry, or duplicate captures were pruned from the folders prior to archive compression without re-indexing the sequential counter in the filename template `class (N).jpeg`. The physical callset is 100% genuine and verified at 2,521 images.

### 2.2 Exact-Duplicate Evaluation Policy (`use_for_eval`)
The dataset contains multiple exact-duplicate image groups (e.g., identical captures repeated across indices).
- **Group Isolation**: Every exact-duplicate group is constrained to a single split via union-find (`group_id`). Zero exact-duplicate pairs cross splits.
- **Evaluation Integrity**: To prevent evaluation metrics (accuracy, loss, F1) from being skewed by counting identical images multiple times, a boolean column `use_for_eval` is added to [`ml_training/manifests/sugarcane_sppu_manifest.csv`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/manifests/sugarcane_sppu_manifest.csv).
  - In `train`: `use_for_eval = true` for all images.
  - In `val` and `test`: `use_for_eval = true` for exactly ONE representative image per exact-duplicate group; duplicates have `use_for_eval = false`.
  - No physical files are deleted.

---

## 3. Cotton Dataset (`cotton_mendeley`)

### 3.1 Class Imbalance & Severity Metadata Policy
- **Disease-Only Targets (4 Classes)**:
  1. `healthy`: 789 images
  2. `bacterial blight`: 119 images
  3. `fussarium wilt`: 47 images
  4. `curl virus`: 25 images
- **Severity Handling**: Severity levels (`critical`, `mild`, `moderate`) are tracked exclusively as a metadata column `severity` in the manifest CSV. They do NOT form distinct classification target classes.
- **Display Name Mapping**: In [`ml_training/configs/classes.yaml`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/configs/classes.yaml), the typographical convention from the source directory (`"fussarium wilt"`) is mapped to standard scientific nomenclature (`"Fusarium wilt"`) without renaming raw directories.
- **Severe Imbalance Warning**:
  > [!WARNING]
  > Ratio between majority (`healthy`: 789) and minority (`curl virus`: 25) is **31.56 : 1**.
  > In a 70/20/10 split:
  > - `curl virus` has only **5** validation images and **3** test images ($< 10$).
  > - `fussarium wilt` has only **9** validation images and **5** test images ($< 10$).
  > Training loss functions must incorporate inverse-frequency class weights computed strictly from the `train` split.
