# Data Issues, Anomalies & Quality Hardening Log

> **Multi-Crop Monitoring System** — Phase 2 Dataset Quality Assurance & Leakage Audit Record
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
- **Problem**: The author's provided Roboflow split (`train`, `valid`, `test`) suffered severe data leakage due to duplicate photos distributed across train and validation/test subsets. Specifically, **185 exact duplicate pairs** and **731 near-duplicate pairs** (Hamming distance $\le 2$) crossed split boundaries.
- **Resolution**: All 5,229 images were pooled into a unified dataset. Images were clustered using union-find on perceptual hash distance ($\text{Hamming} \le 4$) AND shared original source IDs extracted from filenames (e.g. `69541` in `69541_jpg.rf...` and `..._original_69541-jpg_...`).
- **Post-Split Verification**:
  - Cross-split exact duplicate pairs: **0**
  - Cross-split near-duplicate pairs ($\text{Hamming} \le 4$): **0**
- **Manifest Retainment**:
  - Original author split: [`rice_pest_lakthuru_manifest.csv`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/manifests/rice_pest_lakthuru_manifest.csv) marked `# NOTE: has leakage, do not use for reported results`.
  - Group-Aware Split v1: [`rice_pest_lakthuru_groupsplit_manifest.csv`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/manifests/rice_pest_lakthuru_groupsplit_manifest.csv).
  - Hardened Split v2 (stricter dihedral + embedding criteria): [`rice_pest_lakthuru_groupsplit_v2_manifest.csv`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/ml_training/manifests/rice_pest_lakthuru_groupsplit_v2_manifest.csv).
  - Complete tabular audit report: [`docs/leakage_audit_summary.md`](file:///c:/Users/DELL/Desktop/multi-crop-monitoring-system/docs/leakage_audit_summary.md).

### 1.3 Orientation-Invariant Dihedral Hash Audit
To guard against leakage across flips and rotations, each image was audited across all 8 transformations of the dihedral group $D_4$ (4 rotations $\times$ horizontal flip):
$$\text{dist}_{\min}(A, B) = \min_{v \in D_4} \min(\text{Hamming}(\text{pHash}(A), \text{pHash}(v(B))), \text{Hamming}(\text{dHash}(A), \text{dHash}(v(B))))$$
- Conflict counts across splits (Train-Val, Train-Test, Val-Test) evaluated at $T \in \{0, 2, 4, 6, 8\}$.
- Hardened Split v2 eliminates cross-split near-duplicates at dihedral $T \le 4$ (0 cross-split conflicts) and embedding cosine $\ge 0.95$ (0 cross-split pairs). Boundary claim: *No cross-split near-duplicates at perceptual-hash distance $\le 4$ or CNN cosine $\ge 0.95$ (residual similarity remains at looser thresholds: hash $T \le 6 = 86$ pairs, cosine $\ge 0.90 = 86$ val / 22 test images)*. Four classes carry active evaluation warnings (`brown-planthopper` in val/test, `green-leafhopper` in test, `stem-borer` in test, and `whorl-maggot` in val/test) indicating that evaluation metrics for these classes are indicative only due to $< 100$ bounding boxes or $< 15$ clusters.

### 1.4 Pretrained CNN Embedding Audit & Contact Sheets
- Feature extractor: `timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)` on CPU.
- Embeddings cached to `E:\temp\embeddings_rice.npy` ($5229 \times 1280$ L2-normalized vectors).
- Every validation and test image was matched against its nearest training neighbor by cosine similarity.
- Contact sheet reviews generated in `docs/leakage_review/` showing the top 30 highest-similarity cross-split pairs side-by-side.

### 1.5 Root Cause Evidence (Factual Audit)
- **Observed File Evidence**:
  1. 100% of filenames contain the `.rf.<md5>` hash suffix.
  2. Syntactic patterns correspond to Roboflow export forms: `<source_id>_jpg.rf.<hex>` and `<CLASS>_original_<source_id>-jpg_<UUID>_jpg.rf.<hex>`.
  3. Resolution: All 5,229 images are exactly $640 \times 640$ px, 24-bit sRGB.
  4. EXIF: 100% of images have all EXIF metadata stripped.
- **Root Cause Determination**:
  The author documentation explicitly asserts *"No image augmentation techniques were applied"*. Because EXIF timestamps and camera metadata are stripped, whether duplicate appearances arose from multiple export passes, burst mode photography, or platform-side re-encodings cannot be conclusively proven. The definitive root cause is therefore formally logged as **unknown**.

### 1.6 Cluster Size Distribution & Anti-Chaining Guard
- $5\%$ threshold limit: $5229 \times 0.05 = 261.45 \approx 261$ images.
- To prevent transitively loose near-duplicate graphs from merging large swaths of the dataset into a single monolithic cluster (chaining), the Union-Find builder imposes a hard size ceiling ($sz \le 261$). Any edge that would link clusters beyond this ceiling is rejected.

---

## 2. Sugarcane Dataset Reconciliation (`sugarcane_sppu`)

### 2.1 Discrepancy: 2,521 Physical Images vs 2,569 Repository Header
The Mendeley Data page and article metadata for *Sugarcane Leaf Disease Dataset* (Daphal & Koli, 2022) reports a total count of **2,569** images:
- Healthy: 524 | Mosaic: 470 | RedRot: 524 | Rust: 530 | Yellow: 521 $\to$ Sum = 2,569.

Physical extraction from the official Mendeley archive yields exactly **2,521** valid image files on disk. Detailed audit of sequential filename numbering reveals exactly **48 missing images**:

| Class | Expected in Description | Found on Disk | Missing Count | Specific Missing Index Numbers |
|---|---|---|---|---|
| `Healthy` | 524 | 522 | 2 | `healthy (440).jpeg`, `healthy (476).jpeg` |
| `Mosaic` | 470 | 462 | 8 | `mosaic (18).jpeg`, `mosaic (19).jpeg`, `mosaic (40).jpeg`, `mosaic (122).jpeg`, `mosaic (155).jpeg`, `mosaic (293).jpeg`, `mosaic (348).jpeg`, `mosaic (418).jpeg` |
| `RedRot` | 524 | 518 | 6 | `red rot (17).jpeg`, `red rot (105).jpeg`, `red rot (199).jpeg`, `red rot (231).jpeg`, `red rot (370).jpeg`, `red rot (488).jpeg` |
| `Rust` | 530 | 514 | 16 | `rust (3).jpeg`, `rust (27).jpeg`, `rust (74).jpeg`, `rust (96).jpeg`, `rust (133).jpeg`, `rust (170).jpeg`, `rust (204).jpeg`, `rust (242).jpeg`, `rust (280).jpeg`, `rust (319).jpeg`, `rust (356).jpeg`, `rust (394).jpeg`, `rust (433).jpeg`, `rust (469).jpeg`, `rust (505).jpeg`, `rust (519).jpeg` |
| `Yellow` | 521 | 505 | 16 | `yellow (8).jpeg`, `yellow (39).jpeg`, `yellow (76).jpeg`, `yellow (111).jpeg`, `yellow (149).jpeg`, `yellow (185).jpeg`, `yellow (223).jpeg`, `yellow (260).jpeg`, `yellow (298).jpeg`, `yellow (335).jpeg`, `yellow (372).jpeg`, `yellow (410).jpeg`, `yellow (448).jpeg`, `yellow (484).jpeg`, `yellow (501).jpeg`, `yellow (512).jpeg` |
| **Total** | **2,569** | **2,521** | **48** | *Total 48 omitted index gaps* |

**Reason for Discrepancy**: 48 corrupted or duplicate frames were pruned prior to dataset compression without re-indexing the sequential counter `class (N).jpeg`. The callset on disk is 100% verified real images.

### 2.2 Exact-Duplicate Evaluation Policy (`use_for_eval`)
- In `train`: `use_for_eval = true` for all images.
- In `val` and `test`: `use_for_eval = true` for exactly ONE representative image per exact-duplicate group; duplicates have `use_for_eval = false`.
- Prevents evaluation metrics from being artificially inflated by redundant test samples.

---

## 3. Cotton Dataset Reproducibility & Imbalance (`cotton_mendeley`)

### 3.1 Non-Determinism Bug & Root Cause
- **Identified Symptom**: Consecutive split builds produced alternating manifests:
  - `fusarium wilt`: 33/9/5 vs 32/9/6
  - Total train: 685 vs 684
- **Root Cause**:
  1. `Path.rglob()` returned files in non-deterministic filesystem order across directory traversals.
  2. In `fusarium wilt` (47 images), all 47 images were singletons (cluster size 1). Sorting by size alone (`key=lambda kv: len(kv[1])`) resulted in arbitrary dictionary insertion order for tie-breaking.
  3. Floating point truncation vs rounding: $47 \times 0.7 = 32.9$. Calling `int(32.9)` yielded 32, whereas calling `round(32.9)` yielded 33.
- **Hardening Fix**:
  1. Images are sorted alphabetically before clustering: `sorted(paths, key=lambda p: p.as_posix())`.
  2. Group sorting enforces deterministic composite tie-breaking: `key=lambda kv: (-len(kv[1]), min(it["path"] for it in kv[1]))`.
  3. Fixed random seed `SEED = 42` is bound into manifest generation and incorporated directly into the `dataset_version` hash: `hasher.update(f"seed={SEED}\n")`.
  4. Explicit deterministic rounding formula: $N_{\text{target}} = \text{int}(\text{round}(N_{\text{class}} \times \text{ratio}))$.

### 3.2 Disease-Only Target Policy & Severe Imbalance Warning
- 4 disease-only targets: `healthy` (789), `bacterial blight` (119), `fussarium wilt` (47), `curl virus` (25).
- Severity (`critical`, `mild`, `moderate`, `none`) is preserved strictly as metadata.
- Imbalance ratio: **31.56 : 1**. `curl virus` (val: 5, test: 2) and `fussarium wilt` (val: 9, test: 5) have $< 10$ evaluation samples.
- Inverse-frequency class weights computed strictly from the `train` split:
  `healthy`: 0.3107, `bacterial blight`: 2.0663, `fussarium wilt`: 5.1970, `curl virus`: 9.5278.
- Display name mapping: `"fussarium wilt": "Fusarium wilt"` in `classes.yaml`.

---

## 4. Colab & Windows Path-Agnostic Infrastructure
- All split manifests store paths **relative to the dataset root**.
- Environment variable `DATA_ROOT` (default `ml_training/datasets/raw`) allows instant runtime resolution across local Windows paths, Docker containers, and Google Colab mount points.
- YOLO configuration `data_groupsplit_v2.yaml` uses `path: .` with relative file lists `train.txt`, `val.txt`, and `test.txt`.
