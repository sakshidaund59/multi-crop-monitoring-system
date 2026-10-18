# Leakage Audit Summary & Hardened v2 Dataset Split Report

> **Multi-Crop Monitoring System** — Phase 2 Dataset Quality Assurance & Leakage Audit Record  
> *Tables only, all metrics programmatically computed from physical dataset callsets.*

---

## 1. Rice Pest v2 Class & Split Representation

| Pest Class | Split | Images Containing Class | Clean Bounding Boxes | Independent Clusters | Health Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `brown-planthopper` | train | 382 | 1086 | 113 | OK |
| `brown-planthopper` | val | 53 | 357 | 7 | WARN: <15 clusters (7) |
| `brown-planthopper` | test | 70 | 135 | 13 | WARN: <15 clusters (13) |
| `green-leafhopper` | train | 726 | 762 | 161 | OK |
| `green-leafhopper` | val | 212 | 220 | 47 | OK |
| `green-leafhopper` | test | 94 | 95 | 23 | WARN: <100 boxes (95) |
| `leaf-folder` | train | 711 | 1031 | 108 | OK |
| `leaf-folder` | val | 181 | 300 | 27 | OK |
| `leaf-folder` | test | 128 | 128 | 19 | OK |
| `rice-bug` | train | 672 | 916 | 128 | OK |
| `rice-bug` | val | 228 | 266 | 48 | OK |
| `rice-bug` | test | 102 | 112 | 17 | OK |
| `stem-borer` | train | 679 | 758 | 83 | OK |
| `stem-borer` | val | 216 | 220 | 26 | OK |
| `stem-borer` | test | 94 | 94 | 12 | WARN: <100 boxes (94), <15 clusters (12) |
| `whorl-maggot` | train | 450 | 1139 | 32 | OK |
| `whorl-maggot` | val | 138 | 330 | 13 | WARN: <15 clusters (13) |
| `whorl-maggot` | test | 68 | 147 | 3 | WARN: <15 clusters (3) |

---

## 2. Representation Summary Across Splits

| Pest Class | Train Img (Boxes, Clust) | Val Img (Boxes, Clust) | Test Img (Boxes, Clust) | Min Val/Test Boxes | Min Val/Test Clusters | Representation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `brown-planthopper` | 382 (1086 b, 113 c) | 53 (357 b, 7 c) | 70 (135 b, 13 c) | 135 | 7 | **WARNING** |
| `green-leafhopper` | 726 (762 b, 161 c) | 212 (220 b, 47 c) | 94 (95 b, 23 c) | 95 | 23 | **WARNING** |
| `leaf-folder` | 711 (1031 b, 108 c) | 181 (300 b, 27 c) | 128 (128 b, 19 c) | 128 | 19 | **PASSED (Healthy)** |
| `rice-bug` | 672 (916 b, 128 c) | 228 (266 b, 48 c) | 102 (112 b, 17 c) | 112 | 17 | **PASSED (Healthy)** |
| `stem-borer` | 679 (758 b, 83 c) | 216 (220 b, 26 c) | 94 (94 b, 12 c) | 94 | 12 | **WARNING** |
| `whorl-maggot` | 450 (1139 b, 32 c) | 138 (330 b, 13 c) | 68 (147 b, 3 c) | 147 | 3 | **WARNING** |

---

## 3. Orientation-Invariant Dihedral Hash Audit (pHash & dHash)

| Threshold $T$ | v1 Cross-Split Conflicts | v2 Cross-Split Conflicts | Relative Reduction | Leakage Status |
| :---: | :---: | :---: | :---: | :--- |
| $T \le 0$ | 5 | 0 | 100.0% | **PASSED (0 leaks)** |
| $T \le 2$ | 50 | 0 | 100.0% | **PASSED (0 leaks)** |
| $T \le 4$ | 157 | 0 | 100.0% | **PASSED (0 leaks)** |
| $T \le 6$ | 430 | 86 | 80.0% | Reduced (86 left) |
| $T \le 8$ | 1060 | 454 | 57.2% | Reduced (454 left) |

---

## 4. Pretrained CNN Embedding Audit (EfficientNet-B0 Nearest Train Similarity)

| Cosine Similarity Threshold | v1 Validation Images | v2 Validation Images | v1 Test Images | v2 Test Images | Cross-Split Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| Cosine $\ge 0.90$ | 124 (11.9%) | 86 (8.3%) | 78 (14.9%) | 22 (3.9%) | Audited |
| Cosine $\ge 0.95$ | 33 (3.2%) | 0 (0.0%) | 20 (3.8%) | 0 (0.0%) | **PASSED (0 leaks)** |
| Cosine $\ge 0.98$ | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **PASSED (0 leaks)** |

---

## 5. Canonical Source ID Cross-Split Leakage

| Split Version | Total Images | Unique Source IDs | Cross-Split Source IDs | Leakage Status |
| :--- | :---: | :---: | :---: | :--- |
| `CURRENT (v1)` | 5,229 | 1166 | 6 | Leaking |
| `HARDENED (v2)` | 5,229 | 1166 | **0** | **PASSED (0 leaks)** |

---

## 6. Dataset Partition Inventory & Version Register

| Dataset Partition | Version Hash | Train Images | Val Images | Test Images | Total Images | Independent Groups | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Rice Pest v1 (`groupsplit`) | `f4b82c9e7102` | 3,660 | 1,046 | 523 | 5,229 | 1,009 | Residual Leakage |
| Rice Pest v2 (`groupsplit_v2`) | `6b9e5c4ed209` | 3639 | 1033 | 557 | 5,229 | 876 | **Hardened (0 leaks)** |

