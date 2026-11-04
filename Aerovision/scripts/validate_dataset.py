"""
Sanity-check a YOLO-format defect-detection dataset before spending
time training on it.

Run manually:
    python scripts/validate_dataset.py --data dataset/data.yaml

Also imported and called automatically by train.py before training
starts (train.py aborts if validation fails, unless you pass
--skip-validation).

Supports BOTH common data.yaml layouts:
  1. Roboflow's native export layout (default when you download a
     YOLOv8 dataset from Roboflow Universe and DON'T reorganize it):
         dataset/
           train/images/*.jpg   train/labels/*.txt
           valid/images/*.jpg   valid/labels/*.txt
         data.yaml:
           train: train/images
           val: valid/images
           names: [...]
  2. The images/{train,val} + labels/{train,val} layout described in
     AeroVision's own train.py docstring:
         dataset/
           images/train/*.jpg   labels/train/*.txt
           images/val/*.jpg     labels/val/*.txt
         data.yaml:
           path: /absolute/path/to/dataset
           train: images/train
           val: images/val
           names: [...]

Either way, this script resolves train/val image directories exactly
from whatever `train:`/`val:` (and optional `path:`) say in your
data.yaml, then derives the matching labels directory by replacing
"images" with "labels" in that path — the same convention ultralytics
itself uses. No manual folder renaming needed for a Roboflow export.

Checks performed:
  1. Resolved train/val image + label directories exist and are non-empty.
  2. Every image has a matching label file and vice versa (orphans
     are reported, not necessarily fatal — see EXIT CODE below).
  3. Every label file parses correctly: exactly 5 whitespace-separated
     values per line (class_id x_center y_center width height), class_id
     an integer within range of data.yaml's `names`, and all four
     coordinates in [0, 1].
  4. Class distribution — instance counts per class, per split, so a
     severely imbalanced dataset (or an accidentally-empty class) is
     visible before training starts.

EXIT CODE / return value:
  Returns True (pass) only if directories exist+non-empty AND all
  label files parse correctly. Orphaned images/labels and class
  imbalance are printed as WARNINGS, not failures.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve_split_dir(data_yaml_path: str, cfg: dict, key: str) -> Path:
    """
    Resolve a data.yaml `train:`/`val:` entry to an absolute images
    directory, matching how ultralytics itself resolves it:
      - if `path:` is present, the split path is relative to `path:`
        (and `path:` itself is resolved relative to the yaml file if
        it's a relative path; use an absolute `path:` to avoid
        ambiguity — see train.py's docstring note)
      - if `path:` is absent (Roboflow's default export), the split
        path is relative to the yaml file's own directory
    """
    yaml_dir = Path(data_yaml_path).resolve().parent
    split_value = Path(cfg[key])

    if "path" in cfg and cfg["path"]:
        base = Path(cfg["path"]).expanduser()
        if not base.is_absolute():
            base = (yaml_dir / base).resolve()
        return (base / split_value).resolve()

    return (yaml_dir / split_value).resolve()


def _images_dir_to_labels_dir(images_dir: Path) -> Path:
    """Swap the LAST 'images' path segment for 'labels' — same convention ultralytics uses."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    # Fallback: no "images" segment found (unusual layout) — assume a sibling "labels" dir.
    return images_dir.parent / "labels"


def _load_data_yaml(data_yaml_path: str) -> Tuple[Dict[str, Path], Dict[int, str]]:
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    dirs = {}
    for split, key in (("train", "train"), ("val", "val")):
        img_dir = _resolve_split_dir(data_yaml_path, cfg, key)
        dirs[f"images/{split}"] = img_dir
        dirs[f"labels/{split}"] = _images_dir_to_labels_dir(img_dir)

    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}
    return dirs, names


def _check_dirs(dirs: Dict[str, Path]) -> bool:
    ok = True
    print("\n[1/4] Checking directory structure...")
    for label, path in dirs.items():
        if not path.is_dir():
            print(f"  FAIL  {label}: directory not found at {path}")
            ok = False
            continue
        count = sum(1 for p in path.iterdir() if p.is_file())
        if count == 0:
            print(f"  FAIL  {label}: directory exists but is empty ({path})")
            ok = False
        else:
            print(f"  OK    {label}: {count} file(s)  [{path}]")
    return ok


def _stem_set(path: Path, exts: set) -> Dict[str, Path]:
    return {p.stem: p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts}


def _check_orphans(dirs: Dict[str, Path]) -> None:
    print("\n[2/4] Checking for orphaned images/labels...")
    for split in ("train", "val"):
        img_dir, lbl_dir = dirs[f"images/{split}"], dirs[f"labels/{split}"]
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        images = _stem_set(img_dir, IMAGE_EXTS)
        labels = _stem_set(lbl_dir, {".txt"})

        orphan_images = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))

        if orphan_images:
            print(f"  WARN  {split}: {len(orphan_images)} image(s) with no matching label "
                  f"(e.g. {orphan_images[:3]})")
        if orphan_labels:
            print(f"  WARN  {split}: {len(orphan_labels)} label(s) with no matching image "
                  f"(e.g. {orphan_labels[:3]})")
        if not orphan_images and not orphan_labels:
            print(f"  OK    {split}: every image has a label and vice versa "
                  f"({len(images)} pairs)")


def _parse_and_count(dirs: Dict[str, Path], names: Dict[int, str]) -> Tuple[bool, Dict[str, Counter]]:
    print("\n[3/4] Parsing label files...")
    ok = True
    counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    num_classes = len(names)

    for split in ("train", "val"):
        lbl_dir = dirs[f"labels/{split}"]
        if not lbl_dir.is_dir():
            continue
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            with open(lbl_file, "r") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"  FAIL  {lbl_file.name}:{line_no}: expected 5 values, got {len(parts)}")
                        ok = False
                        continue
                    try:
                        cls_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        print(f"  FAIL  {lbl_file.name}:{line_no}: could not parse values: {line!r}")
                        ok = False
                        continue
                    if not (0 <= cls_id < num_classes):
                        print(f"  FAIL  {lbl_file.name}:{line_no}: class_id {cls_id} out of range "
                              f"(expected 0-{num_classes - 1})")
                        ok = False
                        continue
                    if any(not (0.0 <= c <= 1.0) for c in coords):
                        print(f"  FAIL  {lbl_file.name}:{line_no}: coordinates out of [0,1] range: {coords}")
                        ok = False
                        continue
                    counts[split][cls_id] += 1

    if ok:
        print("  OK    all label files parsed without errors")
    return ok, counts


def _print_class_distribution(counts: Dict[str, Counter], names: Dict[int, str]) -> None:
    print("\n[4/4] Class distribution:")
    header = f"  {'class':<15}{'train':>10}{'val':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    max_count = max(
        [c for split_counts in counts.values() for c in split_counts.values()] or [0]
    )
    zero_or_imbalanced = []
    for cls_id, name in sorted(names.items()):
        train_n = counts["train"].get(cls_id, 0)
        val_n = counts["val"].get(cls_id, 0)
        print(f"  {name:<15}{train_n:>10}{val_n:>10}")
        total = train_n + val_n
        if total == 0:
            zero_or_imbalanced.append(f"'{name}' has ZERO instances")
        elif max_count and total < max_count * 0.1:
            zero_or_imbalanced.append(f"'{name}' has far fewer instances than the largest class (possible imbalance)")

    if zero_or_imbalanced:
        print("\n  WARN  Class balance issues:")
        for msg in zero_or_imbalanced:
            print(f"    - {msg}")


def validate_dataset(data_yaml_path: str) -> bool:
    print(f"Validating dataset: {data_yaml_path}")
    dirs, names = _load_data_yaml(data_yaml_path)
    print(f"  Classes ({len(names)}): {names}")

    expected = {"crack", "corrosion", "dent", "rivet_damage"}
    actual = set(names.values())
    if not actual.issubset(expected):
        unexpected = actual - expected
        print(f"\n  WARN  data.yaml has class name(s) {sorted(unexpected)} not in "
              f"AeroVision's expected set {sorted(expected)}. "
              f"Any name not in severity.py's TYPE_CRITICALITY / report.py's COLOR_MAP "
              f"will silently fall back to default styling/criticality downstream.")
    missing = expected - actual
    if missing:
        print(f"  NOTE  data.yaml is missing {sorted(missing)} — that's fine to train "
              f"with a subset of classes, those defect types just won't be detected "
              f"by this model until you add data for them later.")

    dirs_ok = _check_dirs(dirs)
    if not dirs_ok:
        print("\nRESULT: FAIL (missing/empty directories) — fix before training.")
        return False

    _check_orphans(dirs)
    labels_ok, counts = _parse_and_count(dirs, names)
    _print_class_distribution(counts, names)

    print(f"\nRESULT: {'PASS' if labels_ok else 'FAIL'}")
    return labels_ok


def main():
    parser = argparse.ArgumentParser(description="Validate a YOLO-format defect dataset")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    args = parser.parse_args()
    ok = validate_dataset(args.data)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
