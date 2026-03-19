#!/usr/bin/env python3
"""
Train YOLOv8l on the competition dataset.
Run inside the container via step3b_train.slurm
"""

from pathlib import Path

# ================================================================
WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")
YOLO_DATASET_DIR = WORKDIR / "yolo_dataset"
# ================================================================


def main():
    import torch
    from ultralytics import YOLO

    yaml_path = YOLO_DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found!")
        print("Run step3a_convert.slurm first.")
        return

    print("=" * 50)
    print("  Training YOLOv8l")
    print("=" * 50)
    print(f"  Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Dataset: {yaml_path}")
    print()

    model = YOLO("yolov8l.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=80,
        imgsz=1280,
        batch=-1,             # auto batch size
        device=0,

        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.2,
        degrees=5.0,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,

        # Training params
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Saving
        save=True,
        save_period=10,
        plots=True,
        patience=15,
        close_mosaic=10,

        workers=8,
        project=str(WORKDIR / "runs"),
        name="yolov8l_v1",
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  ✓ Training complete!")
    print(f"  Best weights: {best}")
    print(f"  Results dir:  {results.save_dir}")
    print(f"\n  Next: sbatch step3c_package.slurm")


if __name__ == "__main__":
    main()
