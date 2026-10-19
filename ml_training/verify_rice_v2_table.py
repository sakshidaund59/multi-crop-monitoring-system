import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ml_training/datasets/raw/rice_pest_lakthuru")
manifest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "ml_training/manifests/rice_pest_lakthuru_groupsplit_v2_manifest.csv")

names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]

rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
cols = list(rows[0].keys())
print("manifest:", manifest)
print("manifest columns:", cols)
print("example manifest path:", rows[0][cols[0]])


def pick(*keys):
    for k in cols:
        if any(key in k.lower() for key in keys):
            return k
    raise SystemExit(f"no column matching {keys} in {cols}")


c_path, c_split, c_group = pick("path", "file"), pick("split"), pick("group", "cluster")

imgs = defaultdict(lambda: defaultdict(set))
boxes = defaultdict(lambda: defaultdict(int))
groups = defaultdict(lambda: defaultdict(set))
split_images = defaultdict(int)
split_groups = defaultdict(set)
missing = 0
degenerate = 0

for r in rows:
    sp, grp = r[c_split], r[c_group]
    split_images[sp] += 1
    split_groups[sp].add(grp)
    rel = Path(r[c_path].replace("\\", "/"))
    parts = ["labels" if p == "images" else p for p in rel.parts]
    lab = (root / Path(*parts)).with_suffix(".txt")
    if not lab.exists():
        missing += 1
        continue
    for line in lab.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        cls, w = int(float(f[0])), float(f[3])
        if w <= 0:
            degenerate += 1
            continue
        n = names[cls]
        imgs[n][sp].add(rel.as_posix())
        boxes[n][sp] += 1
        groups[n][sp].add(grp)

print(f"\nmanifest rows = {len(rows)}   missing label files = {missing}   "
      f"degenerate boxes skipped (w<=0) = {degenerate}")
for sp in ("train", "val", "test"):
    print(f"{sp:<6} images = {split_images[sp]}   clusters = {len(split_groups[sp])}")

print(f"\n{'class':<20}{'split':<7}{'images':>8}{'boxes':>8}{'clusters':>10}")
for n in names:
    for sp in ("train", "val", "test"):
        print(f"{n:<20}{sp:<7}{len(imgs[n][sp]):>8}{boxes[n][sp]:>8}{len(groups[n][sp]):>10}")