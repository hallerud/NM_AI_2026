#!/usr/bin/env python3
"""
Create YOLO dataset using ALL 248 images for training.
Val set = same images (just for ultralytics to not complain).
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")
ANNOTATIONS_FILE = WORKDIR / "data" / "train" / "annotations.json"
IMAGES_DIR = WORKDIR / "data" / "train" / "images"
YOLO_DATASET_DIR = WORKDIR / "yolo_dataset_full"


def main():
    print("Converting COCO → YOLO (full dataset, no holdout)")

    with open(ANNOTATIONS_FILE) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = data["categories"]
    annotations = data["annotations"]

    print(f"  Images: {len(images)}, Categories: {len(categories)}, Annotations: {len(annotations)}")

    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    all_ids = sorted(images.keys())

    # Use all images for train, and a small subset for val (just for metrics)
    val_ids = all_ids[:10]

    for split_name, split_ids in [("train", all_ids), ("val", val_ids)]:
        img_out = YOLO_DATASET_DIR / split_name / "images"
        lbl_out = YOLO_DATASET_DIR / split_name / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)

        for img_id in split_ids:
            img_info = images[img_id]
            w, h = img_info["width"], img_info["height"]
            fname = img_info["file_name"]

            src = IMAGES_DIR / fname
            if not src.exists():
                continue

            dst = img_out / fname
            if not dst.exists():
                dst.symlink_to(src.resolve())

            lines = []
            for ann in img_anns.get(img_id, []):
                cat_id = ann["category_id"]
                bx, by, bw, bh = ann["bbox"]
                cx = max(0.0, min(1.0, (bx + bw / 2) / w))
                cy = max(0.0, min(1.0, (by + bh / 2) / h))
                nw = max(0.001, min(1.0, bw / w))
                nh = max(0.001, min(1.0, bh / h))
                lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(lbl_out / (Path(fname).stem + ".txt"), "w") as f:
                f.write("\n".join(lines))

    # dataset.yaml
    cat_names = {c["id"]: c["name"] for c in categories}
    yaml_lines = [
        f"path: {YOLO_DATASET_DIR.resolve()}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(categories)}",
        "names:",
    ]
    for cid in sorted(cat_names.keys()):
        name = cat_names[cid].replace("'", "''")
        yaml_lines.append(f"  {cid}: '{name}'")

    with open(YOLO_DATASET_DIR / "dataset.yaml", "w") as f:
        f.write("\n".join(yaml_lines))

    t = len(list((YOLO_DATASET_DIR / "train" / "images").iterdir()))
    v = len(list((YOLO_DATASET_DIR / "val" / "images").iterdir()))
    print(f"  Done: {t} train, {v} val")


if __name__ == "__main__":
    main()
