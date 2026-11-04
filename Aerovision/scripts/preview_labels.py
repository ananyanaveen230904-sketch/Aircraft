"""
Quick visual sanity check: draws the YOLO label boxes onto a handful
of training images and saves them to output/label_preview/, so you
can eyeball whether the labels actually line up with real defects —
useful when training metrics look suspiciously bad and you want to
rule out a labeling/data problem before blaming epochs/hyperparameters.

Run:
    python scripts/preview_labels.py --data dataset/data.yaml --n 12
"""

import argparse
import os
import random

import cv2
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=12, help="Number of sample images to preview")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    yaml_dir = os.path.dirname(os.path.abspath(args.data))
    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    key = "train" if args.split == "train" else "val"
    img_dir = os.path.normpath(os.path.join(yaml_dir, cfg[key]))
    lbl_dir = img_dir.replace("images", "labels")

    out_dir = os.path.join("output", "label_preview")
    os.makedirs(out_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = random.sample(image_files, min(args.n, len(image_files)))

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    for fname in sample:
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Could not read image: {img_path}")
            continue
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    color = colors[cls_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = names[cls_id] if cls_id < len(names) else str(cls_id)
                    cv2.putText(img, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            print(f"  No label file for {fname}")

        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, img)

    print(f"Saved {len(sample)} preview image(s) to {out_dir}/")
    print("Open a few of these and check: do the drawn boxes actually sit on real defects?")


if __name__ == "__main__":
    main()
