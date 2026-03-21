#!/usr/bin/env python3
"""
Build YOLO datasets with proper val split and optional product images.

Creates two datasets:
  yolo_split    — train/val split (for honest evaluation)
  yolo_full     — all shelf images for train, same val (for final submission)

Usage:
  python make_dataset.py                        # shelf images only
  python make_dataset.py --product_images       # shelf + product images in train
  python make_dataset.py --val_frac 0.12        # custom val fraction
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")
ANNOTATIONS = WORKDIR / "data" / "train" / "annotations.json"
IMAGES_DIR = WORKDIR / "data" / "train" / "images"
PRODUCT_DIR = WORKDIR / "data" / "NM_NGD_product_images"
PRODUCT_META = PRODUCT_DIR / "metadata.json"


def load_data():
    with open(ANNOTATIONS) as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = data["categories"]
    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_anns[ann["image_id"]].append(ann)
    return images, categories, img_anns


def stratified_split(images, img_anns, val_frac, seed=42):
    """Pick val images prioritizing rare-class coverage."""
    random.seed(seed)
    all_ids = sorted(images.keys())
    n_val = max(1, round(len(all_ids) * val_frac))

    class_to_imgs = defaultdict(set)
    for img_id in all_ids:
        for ann in img_anns.get(img_id, []):
            class_to_imgs[ann["category_id"]].add(img_id)

    # Rarest classes first
    classes_by_freq = sorted(class_to_imgs, key=lambda c: len(class_to_imgs[c]))

    val_ids = set()
    for cat_id in classes_by_freq:
        if len(val_ids) >= n_val:
            break
        candidates = class_to_imgs[cat_id] - val_ids
        if not candidates:
            continue
        # Pick image covering most classes
        best = max(candidates, key=lambda i: len(
            {a["category_id"] for a in img_anns.get(i, [])}
        ))
        val_ids.add(best)

    # Fill remaining slots randomly
    remaining = [i for i in all_ids if i not in val_ids]
    random.shuffle(remaining)
    while len(val_ids) < n_val:
        val_ids.add(remaining.pop())

    train_ids = sorted(set(all_ids) - val_ids)
    val_ids = sorted(val_ids)

    # Report
    val_cls = {a["category_id"] for i in val_ids for a in img_anns.get(i, [])}
    train_cls = {a["category_id"] for i in train_ids for a in img_anns.get(i, [])}
    all_cls = val_cls | train_cls
    print(f"  Shelf split: {len(train_ids)} train / {len(val_ids)} val")
    print(f"  Val class coverage: {len(val_cls)}/{len(all_cls)}")
    missing = len(all_cls - val_cls)
    if missing:
        print(f"  ({missing} rare classes only in train — unavoidable)")

    return train_ids, val_ids


def write_shelf_images(images, img_anns, img_ids, img_out, lbl_out):
    """Write shelf images + labels in YOLO format."""
    count = 0
    for img_id in img_ids:
        info = images[img_id]
        w, h = info["width"], info["height"]
        fname = info["file_name"]
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
        count += 1
    return count


def write_product_images(categories, img_out, lbl_out):
    """Add product catalog images as single-object training samples.

    Each product image gets a centered bbox covering 80% of the image,
    since products are typically centered with some margin.
    """
    if not PRODUCT_META.exists():
        print("  No product metadata found, skipping")
        return 0

    with open(PRODUCT_META) as f:
        meta = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    count = 0
    for product in meta["products"]:
        if not product["has_images"]:
            continue
        cat_id = cat_name_to_id.get(product["product_name"])
        if cat_id is None:
            continue

        product_dir = PRODUCT_DIR / product["product_code"]
        if not product_dir.exists():
            continue

        # Only use front view — keeps dataset size manageable
        # and front is most relevant for shelf detection
        for img_type in ["front"]:
            src = product_dir / f"{img_type}.jpg"
            if not src.exists():
                continue

            fname = f"prod_{product['product_code']}_{img_type}.jpg"
            dst = img_out / fname
            if not dst.exists():
                dst.symlink_to(src.resolve())

            # Centered bbox covering 80% of image
            label = f"{cat_id} 0.500000 0.500000 0.800000 0.800000"
            with open(lbl_out / f"prod_{product['product_code']}_{img_type}.txt", "w") as f:
                f.write(label)
            count += 1

    return count


def write_dataset_yaml(out_dir, categories):
    cat_names = {c["id"]: c["name"] for c in categories}
    lines = [
        f"path: {out_dir.resolve()}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(categories)}",
        "names:",
    ]
    for cid in sorted(cat_names):
        name = cat_names[cid].replace("'", "''")
        lines.append(f"  {cid}: '{name}'")
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write("\n".join(lines))


def build_dataset(out_dir, train_ids, val_ids, images, img_anns, categories,
                  include_products=False):
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = out_dir / split / "images"
        lbl_out = out_dir / split / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)

        n_shelf = write_shelf_images(images, img_anns, ids, img_out, lbl_out)

        n_prod = 0
        if include_products and split == "train":
            n_prod = write_product_images(categories, img_out, lbl_out)

        print(f"  {split}: {n_shelf} shelf" + (f" + {n_prod} product" if n_prod else ""))

    write_dataset_yaml(out_dir, categories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--product_images", action="store_true",
                        help="Add product catalog images to training set")
    args = parser.parse_args()

    print("=" * 50)
    print("  Building YOLO datasets")
    print("=" * 50)

    images, categories, img_anns = load_data()
    print(f"\n  Source: {len(images)} images, {len(categories)} classes")

    train_ids, val_ids = stratified_split(images, img_anns, args.val_frac)

    # Split dataset (honest eval)
    print(f"\n  → yolo_split (for evaluation)")
    build_dataset(
        WORKDIR / "yolo_split", train_ids, val_ids,
        images, img_anns, categories,
        include_products=args.product_images,
    )

    # Full dataset (final submission training)
    all_ids = sorted(images.keys())
    print(f"\n  → yolo_full (for final training, all {len(all_ids)} shelf images)")
    build_dataset(
        WORKDIR / "yolo_full", all_ids, val_ids,
        images, img_anns, categories,
        include_products=args.product_images,
    )

    print(f"\n{'=' * 50}")
    print(f"  Done! Use --dataset yolo_split for eval runs")
    print(f"              --dataset yolo_full  for final runs")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
