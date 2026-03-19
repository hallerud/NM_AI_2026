#!/usr/bin/env python3
"""
Train a YOLO model with config passed via CLI args.
Usage: python sweep_train.py --name exp_name --model yolov8l.pt --imgsz 1280 --epochs 150
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name (used as run dir)")
    parser.add_argument("--model", default="yolov8l.pt", help="Model weights")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=-1, help="-1 for auto")
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--copy_paste", type=float, default=0.2)
    parser.add_argument("--close_mosaic", type=int, default=10)
    parser.add_argument("--dataset", default="yolo_dataset_full", help="Dataset dir name")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    workdir = Path("/cluster/home/ksv023/NM_AI_2026/task1")
    yaml_path = workdir / args.dataset / "dataset.yaml"

    print(f"{'='*50}")
    print(f"  Experiment: {args.name}")
    print(f"  Model: {args.model} | imgsz: {args.imgsz} | epochs: {args.epochs}")
    print(f"  Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Dataset: {yaml_path}")
    print(f"{'='*50}\n")

    model = YOLO(args.model)

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,

        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=args.copy_paste,
        degrees=5.0,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,

        # Training
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,

        # Loss
        box=args.box,
        cls=args.cls,
        dfl=1.5,

        # Saving
        save=True,
        save_period=20,
        plots=True,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        workers=8,

        project=str(workdir / "runs"),
        name=args.name,
    )

    print(f"\n  Done: {args.name}")


if __name__ == "__main__":
    main()
