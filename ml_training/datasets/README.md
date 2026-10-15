# Real Dataset Directory Guide

This directory holds **real-world agricultural image datasets** for training and evaluating vision models across the 6 supported crops (rice, wheat, maize, cotton, soybean, sugarcane).

> **Important Global Policy**: Never synthesize, fabricate, or place placeholder images here. Only real-world imagery with legitimate provenance is accepted.

---

## Directory Organization

```
ml_training/datasets/
├── README.md               # This documentation
├── raw/                    # Put unzipped downloaded raw datasets here (git-ignored)
│   └── <dataset_name>/     # Individual raw dataset root
└── processed/              # Group-aware train/val/test splits & manifests (git-ignored)
    └── <dataset_name>/     # Processed splits with manifest.csv
```

---

## Supported Dataset Layouts

`validate_dataset.py` automatically detects and verifies any of the following 3 standard layouts:

### 1. Classification Layout (Folder-per-Class)
Images are organized into subdirectories named after their respective disease or pest class:
```
ml_training/datasets/raw/<dataset_name>/
├── rice_blast/
│   ├── image_001.jpg
│   └── image_002.jpg
├── brown_spot/
│   ├── image_003.jpg
│   └── image_004.jpg
└── healthy/
    ├── image_005.jpg
    └── image_006.jpg
```
*Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.*

### 2. YOLO Object Detection Layout
Contains images and normalized bounding box coordinates:
```
ml_training/datasets/raw/<dataset_name>/
├── data.yaml               # Optional YOLO config with names/nc
├── images/                 # All image files
│   ├── scan_001.jpg
│   └── scan_002.jpg
└── labels/                 # Matching label text files
    ├── scan_001.txt
    └── scan_002.txt
```
*(Also supports split subdirectories like `train/images`, `train/labels`, `val/images`, etc.)*

**YOLO Label Format (`.txt` per image):**
```
<class_id> <x_center> <y_center> <width> <height>
```
*All coordinates are normalized floats between `0.0` and `1.0`.*

### 3. Pascal VOC XML Detection Layout
Contains images alongside XML bounding box annotations:
```
ml_training/datasets/raw/<dataset_name>/
├── JPEGImages/             # Image files
│   ├── field_001.jpg
│   └── field_002.jpg
└── Annotations/            # XML annotation files
    ├── field_001.xml
    └── field_002.xml
```
*(Flat layout with `.xml` files adjacent to `.jpg` files in the same directory is also supported).*

**Pascal VOC Bounding Box Format (`.xml`):**
```xml
<annotation>
    <size>
        <width>1920</width>
        <height>1080</height>
    </size>
    <object>
        <name>fall_armyworm</name>
        <bndbox>
            <xmin>120</xmin>
            <ymin>80</ymin>
            <xmax>340</xmax>
            <ymax>290</ymax>
        </bndbox>
    </object>
</annotation>
```

---

## Workflow Steps

1. **Download real dataset** and unpack it into `ml_training/datasets/raw/<dataset_name>/`.
2. **Add metadata & license** in `ml_training/configs/dataset_sources.yaml`.
3. **Run Validation**:
   ```bash
   python ml_training/scripts/validate_dataset.py --dataset <dataset_name>
   ```
4. **Build Group-Aware Splits (70/20/10)**:
   ```bash
   python ml_training/scripts/build_splits.py --dataset <dataset_name>
   ```
5. **Generate Documentation & Statistics**:
   ```bash
   python ml_training/scripts/dataset_report.py
   ```
