#!/usr/bin/env python3
"""
Converts COCO annotations → YOLO format.
Run inside the container via step3a_convert.slurm
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ================================================================
# UPDATE THESE PATHS after running step1 and checking its output!
# ================================================================
WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")

ANNOTATIONS_FILE = WORKDIR / "data" / "train" / "annotations.json"
IMAGES_DIR = WORKDIR / "data" / "train" / "images"

YOLO_DATASET_DIR = WORKDIR / "yolo_dataset"
# ================================================================


def find_annotations():
    """Try to find annotations.json automatically."""
    if ANNOTATIONS_FILE.exists():
        return ANNOTATIONS_FILE
    candidates = list(WORKDIR.rglob("annotations.json"))
    if candidates:
        print(f"  Auto-found: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot find annotations.json!\n"
        f"  Looked in: {ANNOTATIONS_FILE}\n"
        f"  Run: find {WORKDIR} -name 'annotations.json'"
    )


def find_images():
    """Try to find the images directory automatically."""
    if IMAGES_DIR.exists() and list(IMAGES_DIR.glob("img_*")):
        return IMAGES_DIR
    # Search for images
    sample = list(WORKDIR.rglob("img_00001.jpg")) + list(WORKDIR.rglob("img_00001.jpeg"))
    if sample:
        img_dir = sample[0].parent
        print(f"  Auto-found images at: {img_dir}")
        return img_dir
    raise FileNotFoundError(
        f"Cannot find images!\n"
        f"  Looked in: {IMAGES_DIR}\n"
        f"  Run: find {WORKDIR} -name 'img_*.jpg' | head -5"
    )


def main():
    print("=" * 50)
    print("  Converting COCO → YOLO format")
    print("=" * 50)

    ann_file = find_annotations()
    img_dir = find_images()
    print(f"\n  Annotations: {ann_file}")
    print(f"  Images:      {img_dir}")

    with open(ann_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = data["categories"]
    annotations = data["annotations"]
    num_classes = len(categories)

    print(f"  Images: {len(images)}, Categories: {num_classes}, Annotations: {len(annotations)}")

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    # 90/10 split
    random.seed(42)
    img_ids = sorted(images.keys())
    random.shuffle(img_ids)
    split_idx = int(0.9 * len(img_ids))
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])
    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val")

    # Clean output
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    total_labels = 0
    missing = 0

    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        img_out = YOLO_DATASET_DIR / split_name / "images"
        lbl_out = YOLO_DATASET_DIR / split_name / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)

        for img_id in split_ids:
            img_info = images[img_id]
            w, h = img_info["width"], img_info["height"]
            fname = img_info["file_name"]

            src = img_dir / fname
            if not src.exists():
                missing += 1
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
            total_labels += len(lines)

    if missing:
        print(f"  ⚠ {missing} images not found")

    print(f"  Labels written: {total_labels}")

    # Create dataset.yaml
    cat_names = {c["id"]: c["name"] for c in categories}
    yaml_lines = [
        f"path: {YOLO_DATASET_DIR.resolve()}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {num_classes}",
        "names:",
    ]
    for cid in sorted(cat_names.keys()):
        name = cat_names[cid].replace("'", "''")
        yaml_lines.append(f"  {cid}: '{name}'")

    yaml_path = YOLO_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    # Verify
    t_imgs = len(list((YOLO_DATASET_DIR / "train" / "images").iterdir()))
    v_imgs = len(list((YOLO_DATASET_DIR / "val" / "images").iterdir()))
    print(f"\n  ✓ Dataset ready: {YOLO_DATASET_DIR}")
    print(f"    Train: {t_imgs} images")
    print(f"    Val:   {v_imgs} images")
    print(f"    Config: {yaml_path}")


if __name__ == "__main__":
    main()
