# Agricultural Dataset Registry & Provenance Report

> **Multi-Crop Monitoring System** - Automated Dataset Inventory & Health Report
> *Enforces zero fake data policy: All figures reflect verified physical image callsets.*

---

## Status: Awaiting Real-World Dataset Placement

No processed split manifests currently exist in `ml_training/manifests/`.

### Supported Crops
The vision pipeline is configured to monitor **6 priority crops**:
1. **Rice** (*Oryza sativa*)
2. **Wheat** (*Triticum aestivum*)
3. **Maize** (*Zea mays*)
4. **Cotton** (*Gossypium hirsutum*)
5. **Soybean** (*Glycine max*)
6. **Sugarcane** (*Saccharum officinarum*)

---

## Instructions to Ingest Real Datasets

1. Download real-world datasets for any of the above crops and place them into:
   ```
   ml_training/datasets/raw/<dataset_name>/
   ```
   Supported layouts:
   - **Classification**: Folders per class containing images (e.g., `ml_training/datasets/raw/rice_leaves/blast/*.jpg`).
   - **YOLO Detection**: `images/` + `labels/*.txt` (normalized bboxes) + optional `data.yaml`.
   - **Pascal VOC XML**: `Annotations/*.xml` + `JPEGImages/*.jpg`.

2. Add dataset source and licensing info to `ml_training/configs/dataset_sources.yaml`.

3. Run dataset validation:
   ```bash
   python ml_training/scripts/validate_dataset.py --dataset <dataset_name>
   ```

4. Build group-aware 70/20/10 train/val/test splits:
   ```bash
   python ml_training/scripts/build_splits.py --dataset <dataset_name>
   ```

5. Update this report with real metrics:
   ```bash
   python ml_training/scripts/dataset_report.py
   ```
