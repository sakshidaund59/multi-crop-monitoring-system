# ML Training & Real Dataset Pipeline (Phase 2)

Automated, rigorous dataset ingestion, validation, and group-aware splitting for the **Multi-Crop Monitoring System**.

---

## Target Crops
1. **Rice** (*Oryza sativa*)
2. **Wheat** (*Triticum aestivum*)
3. **Maize** (*Zea mays*)
4. **Cotton** (*Gossypium hirsutum*)
5. **Soybean** (*Glycine max*)
6. **Sugarcane** (*Saccharum officinarum*)

---

## Structure

```
ml_training/
├── configs/
│   ├── classes.yaml             # Crop class registry (empty-by-default; populated only with real detected classes)
│   └── dataset_sources.yaml     # Dataset provenance, citations, and licenses
├── datasets/
│   ├── README.md                # Guide on expected raw/processed layouts
│   ├── raw/                     # Real downloaded datasets go here (git-ignored)
│   └── processed/               # Processed outputs (git-ignored)
├── manifests/                   # Versioned train/val/test CSV manifests
├── reports/                     # Automated JSON validation reports
├── scripts/
│   ├── validate_dataset.py      # Quality & bbox validation across 3 layouts
│   ├── build_splits.py          # Group-aware 70/20/10 splitting
│   └── dataset_report.py        # Automated docs/dataset.md generator
├── tests/
│   └── test_dataset_pipeline.py # Pytest test suite using in-memory test fixtures
└── requirements.txt             # ML pipeline dependencies
```

---

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r ml_training/requirements.txt
```

### 2. Place Real Datasets
Unpack downloaded datasets into `ml_training/datasets/raw/<dataset_name>/`:
- **Classification**: Folders per class (`<dataset_name>/<class_name>/*.jpg`)
- **YOLO Detection**: `<dataset_name>/images/` + `<dataset_name>/labels/*.txt` + `data.yaml`
- **Pascal VOC XML**: `<dataset_name>/Annotations/*.xml` + `<dataset_name>/JPEGImages/*.jpg`

### 3. Validate Dataset Integrity
```bash
python ml_training/scripts/validate_dataset.py --dataset <dataset_name> --crop <crop_name>
```
*Outputs JSON report to `ml_training/reports/<dataset_name>_validation.json`.*

### 4. Build Group-Aware Splits (70/20/10)
```bash
python ml_training/scripts/build_splits.py --dataset <dataset_name>
```
*Outputs versioned CSV manifest to `ml_training/manifests/<dataset_name>_manifest.csv`.*

### 5. Generate Documentation
```bash
python ml_training/scripts/dataset_report.py
```
*Generates real metrics, class breakdowns, and licensing tables in `docs/dataset.md`.*

---

## Running on Google Colab

```python
# Clone and enter repo
!git clone https://github.com/<your-repo>/multi-crop-monitoring-system.git
%cd multi-crop-monitoring-system

# Install dependencies
!pip install -r ml_training/requirements.txt

# Download your dataset (e.g., from Kaggle / Roboflow / Zenodo / Google Drive)
!mkdir -p ml_training/datasets/raw/rice_leaf_diseases
!unzip /content/drive/MyDrive/rice_leaf_diseases.zip -d ml_training/datasets/raw/rice_leaf_diseases

# Validate and Split
!python ml_training/scripts/validate_dataset.py --dataset rice_leaf_diseases --crop rice
!python ml_training/scripts/build_splits.py --dataset rice_leaf_diseases
!python ml_training/scripts/dataset_report.py
```

---

## Running Tests
```bash
pytest ml_training/tests/ -v
```
