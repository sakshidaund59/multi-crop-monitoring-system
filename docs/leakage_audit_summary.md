# Leakage Audit Summary & Hardened v2 Dataset Split Report

> **Multi-Crop Monitoring System** — Phase 2 Dataset Quality Assurance & Leakage Audit Record  
> *Tables only, all metrics programmatically computed from physical dataset callsets.*

---

## 1. Rice Pest v2 Class & Split Representation

| Pest Class | Split | Images Containing Class | Clean Bounding Boxes | Independent Clusters | Split Health & Diversity Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `brown-planthopper` | train | 382 | 1,086 | 113 | Healthy |
| `brown-planthopper` | val | 53 | 357 | 7 | **WARNING**: < 15 clusters (7 clusters) — Indicative only |
| `brown-planthopper` | test | 70 | 135 | 13 | **WARNING**: < 15 clusters (13 clusters) — Indicative only |
| `green-leafhopper` | train | 726 | 762 | 161 | Healthy |
| `green-leafhopper` | val | 212 | 220 | 47 | Healthy ($\ge 100$ boxes, $\ge 15$ clusters) |
| `green-leafhopper` | test | 94 | 95 | 23 | **WARNING**: < 100 boxes (95 boxes) — Indicative only |
| `leaf-folder` | train | 711 | 1,031 | 108 | Healthy |
| `leaf-folder` | val | 181 | 300 | 27 | Healthy ($\ge 100$ boxes, $\ge 15$ clusters) |
| `leaf-folder` | test | 128 | 128 | 19 | Healthy ($\ge 100$ boxes, $\ge 15$ clusters) |
| `rice-bug` | train | 672 | 916 | 128 | Healthy |
| `rice-bug` | val | 228 | 266 | 48 | Healthy ($\ge 100$ boxes, $\ge 15$ clusters) |
| `rice-bug` | test | 102 | 112 | 17 | Healthy ($\ge 100$ boxes, $\ge 15$ clusters) |
| `stem-borer` | train | 679 | 758 | 83 | Healthy |
| `stem-borer` | val | 216 | 220 | 26 | Healthy ($\ge 100$ boxes, $\ge 15$ clusters) |
| `stem-borer` | test | 94 | 94 | 12 | **WARNING**: < 100 boxes (94 boxes) & < 15 clusters (12 clusters) — Indicative only |
| `whorl-maggot` | train | 450 | 1,139 | 32 | Healthy |
| `whorl-maggot` | val | 138 | 330 | 13 | **WARNING**: < 15 clusters (13 clusters) — Indicative only |
| `whorl-maggot` | test | 68 | 147 | 3 | **WARNING**: < 15 clusters (3 clusters) — Indicative only |

> [!NOTE]
> Metrics for classes with active warnings (`brown-planthopper`, `green-leafhopper`, `stem-borer`, `whorl-maggot`) are indicative only due to bounding box counts < 100 or independent cluster counts < 15 in those specific evaluation partitions. Two classes (`leaf-folder`, `rice-bug`) satisfy both $\ge 100$ boxes and $\ge 15$ clusters across all evaluation splits.

---

## 2. Representation Summary Across Splits

| Pest Class | Train Img (Boxes, Clust) | Val Img (Boxes, Clust) | Test Img (Boxes, Clust) | Min Val/Test Boxes | Min Val/Test Clusters | Evaluation Health Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `brown-planthopper` | 382 (1,086 b, 113 c) | 53 (357 b, 7 c) | 70 (135 b, 13 c) | 135 | 7 | **WARNING** (< 15 clusters in val/test) — Indicative only |
| `green-leafhopper` | 726 (762 b, 161 c) | 212 (220 b, 47 c) | 94 (95 b, 23 c) | 95 | 23 | **WARNING** (< 100 boxes in test) — Indicative only |
| `leaf-folder` | 711 (1,031 b, 108 c) | 181 (300 b, 27 c) | 128 (128 b, 19 c) | 128 | 19 | **PASSED** (Healthy representation) |
| `rice-bug` | 672 (916 b, 128 c) | 228 (266 b, 48 c) | 102 (112 b, 17 c) | 112 | 17 | **PASSED** (Healthy representation) |
| `stem-borer` | 679 (758 b, 83 c) | 216 (220 b, 26 c) | 94 (94 b, 12 c) | 94 | 12 | **WARNING** (< 100 boxes & < 15 clusters in test) — Indicative only |
| `whorl-maggot` | 450 (1,139 b, 32 c) | 138 (330 b, 13 c) | 68 (147 b, 3 c) | 147 | 3 | **WARNING** (< 15 clusters in val/test) — Indicative only |

---

## 3. Orientation-Invariant Dihedral Hash Audit (pHash & dHash)

No cross-split near-duplicates at perceptual-hash distance $\le 4$ or CNN cosine $\ge 0.95$ (residual similarity remains at looser thresholds: hash $T \le 6 = 86$ pairs, cosine $\ge 0.90 = 86$ val / 22 test images).

| Threshold $T$ | v1 Cross-Split Conflicts | v2 Cross-Split Conflicts | Relative Reduction | Audit Status |
| :---: | :---: | :---: | :---: | :--- |
| $T \le 0$ (Exact) | 5 | 0 | 100.0% | **PASSED (0 conflicts at this threshold)** |
| $T \le 2$ | 50 | 0 | 100.0% | **PASSED (0 conflicts at this threshold)** |
| $T \le 4$ | 157 | 0 | 100.0% | **PASSED (0 conflicts at this threshold)** |
| $T \le 6$ | 430 | 86 | 80.0% | Hardened (86 residual pairs) |
| $T \le 8$ | 1,060 | 454 | 57.2% | Hardened (454 residual pairs) |

---

## 4. Pretrained CNN Embedding Audit (EfficientNet-B0 Nearest Train Similarity)

| Cosine Similarity Threshold | v1 Validation Images | v2 Validation Images | v1 Test Images | v2 Test Images | Cross-Split Audit Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| Cosine $\ge 0.90$ | 124 (11.9%) | 86 (8.3%) | 78 (14.9%) | 22 (3.9%) | Residual similarity audited |
| Cosine $\ge 0.95$ | 33 (3.2%) | 0 (0.0%) | 20 (3.8%) | 0 (0.0%) | **PASSED (0 conflicts at this threshold)** |
| Cosine $\ge 0.98$ | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **PASSED (0 conflicts at this threshold)** |

---

## 5. Canonical Source ID Cross-Split Leakage

| Split Version | Total Images | Unique Source IDs | Cross-Split Source IDs | Split Audit Status |
| :--- | :---: | :---: | :---: | :--- |
| `CURRENT (v1)` | 5,229 | 1,166 | 6 | Cross-split leakage detected |
| `HARDENED (v2)` | 5,229 | 1,166 | **0** | **PASSED (0 conflicts at this threshold)** |

---

## 6. Dataset Partition Inventory & Version Register

| Dataset Partition | Version Hash | Train Images | Val Images | Test Images | Total Images | Independent Groups | Split Characterization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Rice Pest v1 (`groupsplit`) | `e35b6f785e0e` | 3,660 | 1,045 | 524 | 5,229 | 1,009 | Residual Leakage ($T \le 4$: 157 pairs) |
| Rice Pest v2 (`groupsplit_v2`) | `6b9e5c4ed209` | 3,639 | 1,033 | 557 | 5,229 | 876 | Hardened: No near-duplicates at pHash $\le 4$ or Cosine $\ge 0.95$ |
| Cotton Mendeley (`reproducible`) | `99c7fa25ceeb` | 686 | 196 | 98 | 980 | 980 | Hardened: Byte-identical reproducibility across runs |
| Wheat Disease (`zenodo_jic`) | `d5dbead715ee` | 697 | 197 | 105 | 999 | 999 | Group-aware partition (no pHash $\le 3$ leaks) |
| Sugarcane SPPU (`mendeley`) | `c2396e678992` | 1,762 | 502 | 257 | 2,521 | 2,294 | Group-aware partition (`use_for_eval` active) |


---

## 7. Cotton Mendeley Dihedral & Embedding Audit

| Threshold $T$ | Cross-Split Conflicts | Audit Status |
| :---: | :---: | :--- |
| $T \le 0$ | 0 | **PASSED (0 conflicts at this threshold)** |
| $T \le 2$ | 0 | **PASSED (0 conflicts at this threshold)** |
| $T \le 4$ | 0 | **PASSED (0 conflicts at this threshold)** |
| $T \le 6$ | 0 | **PASSED (0 conflicts at this threshold)** |
| $T \le 8$ | 0 | **PASSED (0 conflicts at this threshold)** |

---

## 8. Cotton Mendeley Nearest Train Embedding Similarity (EfficientNet-B0)

| Cosine Similarity Threshold | Validation Images | Test Images | Cross-Split Audit Status |
| :---: | :---: | :---: | :--- |
| Cosine $\ge 0.90$ | 9 (4.6%) | 4 (4.1%) | Residual similarity audited |
| Cosine $\ge 0.95$ | 0 (0.0%) | 0 (0.0%) | **PASSED (0 conflicts at this threshold)** |
| Cosine $\ge 0.98$ | 0 (0.0%) | 0 (0.0%) | **PASSED (0 conflicts at this threshold)** |

---

## 9. Cotton Mendeley Class Breakdown & Split Weights

| Disease Class | Total Images | Train | Val | Test | Train Split Weight $w_c$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bacterial blight` | 119 | 83 | 24 | 12 | **2.0663** |
| `curl virus` | 25 | 18 | 5 | 2 | **9.5278** |
| `fussarium wilt` | 47 | 33 | 9 | 5 | **5.1970** |
| `healthy` | 789 | 552 | 158 | 79 | **0.3107** |

