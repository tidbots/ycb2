#!/usr/bin/env python3
"""
coco_to_yolo_split.py

Convert a COCO annotation JSON (bbox) into a YOLO detection dataset with
train/val split.

Expected inputs:
- --coco: path to coco_annotations.json
- --image_dir: directory where image files referenced by COCO "file_name" exist
- --out_dir: output YOLO dataset root
- --train_ratio: fraction for train split (default 0.9)

Outputs:
out_dir/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

Notes:
- YOLO class indices are assigned by sorting COCO category IDs.
- Bounding boxes are converted from COCO xywh (pixels) to YOLO xywh (normalized).
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

import yaml


def coco_to_yolo_bbox(bbox_xywh, w, h):
    """COCO bbox [x,y,w,h] in pixels -> YOLO [xc,yc,w,h] normalized."""
    x, y, bw, bh = bbox_xywh
    xc = (x + bw / 2.0) / float(w)
    yc = (y + bh / 2.0) / float(h)
    bw = bw / float(w)
    bh = bh / float(h)
    return xc, yc, bw, bh


def clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def safe_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path to COCO annotations.json")
    ap.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing images referenced by COCO image file_name",
    )
    ap.add_argument("--out_dir", required=True, help="Output YOLO dataset root")
    ap.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for split")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.coco, "r") as f:
        coco = json.load(f)

    # Map category_id -> name
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    if not cat_id_to_name:
        raise ValueError("COCO JSON has no categories")

    # Assign YOLO class index by sorted category_id for stability
    cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_cls = {cid: i for i, cid in enumerate(cat_ids)}
    names = [cat_id_to_name[cid] for cid in cat_ids]

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    if not images:
        raise ValueError("COCO JSON has no images")

    # Group annotations by image_id
    img_id_to_anns = defaultdict(list)
    for a in anns:
        # skip invalid boxes
        bbox = a.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        if a.get("category_id") not in cat_id_to_cls:
            continue
        img_id_to_anns[a["image_id"]].append(a)

    # Split
    img_ids = [im["id"] for im in images]
    random.shuffle(img_ids)
    n_train = int(len(img_ids) * args.train_ratio)
    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train:])

    out_dir = os.path.abspath(args.out_dir)

    def ensure_dirs(split: str):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    ensure_dirs("train")
    ensure_dirs("val")

    id_to_img = {im["id"]: im for im in images}

    missing_images = 0
    empty_labels = 0

    for img_id in img_ids:
        im = id_to_img[img_id]
        split = "train" if img_id in train_ids else "val"

        file_name = im.get("file_name")
        if not file_name:
            continue

        src = os.path.join(os.path.abspath(args.image_dir), file_name)
        if not os.path.exists(src):
            # Sometimes COCO file_name might be just basename; try that as fallback
            src2 = os.path.join(os.path.abspath(args.image_dir), os.path.basename(file_name))
            if os.path.exists(src2):
                src = src2
            else:
                missing_images += 1
                continue

        dst = os.path.join(out_dir, "images", split, os.path.basename(file_name))
        safe_copy(src, dst)

        w = int(im.get("width", 0))
        h = int(im.get("height", 0))
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size in COCO: id={img_id} width={w} height={h}")

        label_path = os.path.join(
            out_dir,
            "labels",
            split,
            os.path.splitext(os.path.basename(file_name))[0] + ".txt",
        )

        lines = []
        for a in img_id_to_anns.get(img_id, []):
            cid = a["category_id"]
            cls = cat_id_to_cls[cid]
            xc, yc, bw, bh = coco_to_yolo_bbox(a["bbox"], w, h)

            # Clamp to [0,1] to be safe
            xc = clamp01(xc)
            yc = clamp01(yc)
            bw = clamp01(bw)
            bh = clamp01(bh)

            # Skip degenerate boxes
            if bw <= 0.0 or bh <= 0.0:
                continue

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            empty_labels += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    data_yaml = {
        "path": out_dir,
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print("Wrote:", os.path.join(out_dir, "data.yaml"))
    print("Classes:", names)
    print("Train images:", len(train_ids), "Val images:", len(val_ids))
    if missing_images:
        print("WARNING: missing images:", missing_images)
    if empty_labels:
        print("NOTE: images with empty labels:", empty_labels)


if __name__ == "__main__":
    main()

