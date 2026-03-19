# NorgesGruppen Object Detection Competition - Task 1

## Current Status
- **Baseline Score: mAP 0.7615** (first submission)
- Phase 1 complete. Pipeline working end-to-end.
- **Phase 2 sweep running** (6 experiments, submitted 2026-03-19)
- **Deadline: ~2026-03-21** (~65 hours from Phase 2 start)
- **Submissions: 3 per day** (1 used today, 2 remaining)

## Competition Details

### Scoring
- **70% detection mAP** — did you find the products? (IoU >= 0.5, category ignored)
- **30% classification mAP** — correct product ID? (IoU >= 0.5 AND correct category_id)

### Dataset
- 248 training images (shelf images with grocery products)
- 356 product categories + 1 unknown_product (357 total)
- 22,731 annotations (avg ~92 annotations per image — very dense scenes)
- Images are 2000x1500 resolution
- Also available: `NM_NGD_product_images/` — individual product reference images organized by barcode (left, right, front, back, top, main views)

### Sandbox Constraints
| Resource | Limit |
|----------|-------|
| GPU | NVIDIA L4 (24 GB VRAM) |
| CPU | 4 vCPU |
| Memory | 8 GB |
| Timeout | 300 seconds |
| Network | None (offline) |
| Max zip size | 420 MB (uncompressed) |
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| ultralytics | 8.1.0 |
| onnxruntime-gpu | 1.20.0 |

### Security Restrictions
Blocked imports: `os`, `sys`, `subprocess`, `socket`, `shutil`, `pickle`, `yaml`, `requests`, `multiprocessing`, `threading`, etc.
Use `pathlib` for file ops, `json` for config. ONNX format recommended for model weights.

## Infrastructure

### Cluster (Olivia)
- **GPU nodes (accel partition):** ARM64 (Neoverse V2) + 4x GH200 (96 GB HBM3 each)
- **Container:** `/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif`
- **Account:** nn11127k (~986 billing hours, ~1000 GPU hours remaining, expires 2026-04-01)
- **Working dir:** `/cluster/home/ksv023/NM_AI_2026/task1/`

### Environment Setup (solved issues)
- Container ships numpy 2.4.3 but torch compiled against numpy 1.x — **fix:** install numpy 1.26.4 to `pylibs/` and prepend to `PYTHONPATH`
- Container has opencv-python (GUI) but no libxcb — **fix:** uninstall and install opencv-python-headless
- Trained with ultralytics 8.4.24, sandbox has 8.1.0 — **fix:** export to ONNX (opset 17, FP16)
- All slurm scripts use venv + PYTHONPATH override pattern

### Key Paths
```
task1/
├── data/train/annotations.json          # COCO format annotations
├── data/train/images/                   # 248 training images (img_00001.jpg ... img_00248.jpg)
├── data/NM_NGD_product_images/          # Product reference images by barcode
├── yolo_dataset/                        # YOLO format (223 train, 25 val) — Phase 1
├── yolo_dataset_full/                   # YOLO format (248 train, 10 val) — Phase 2
├── runs/yolov8l_v13/                    # Phase 1 best model
├── runs/v20_full_l_1280/                # Phase 2 sweep experiments
├── runs/v21_full_l_1600/
├── runs/v22_full_x_1280/
├── runs/v23_full_x_1600/
├── runs/v24_full_l_highbox/
├── runs/v25_full_l_moreaug/
├── trained_submission/submission.zip     # Current submission (77.5 MB)
├── venv/                                # Python venv (system-site-packages)
├── pylibs/                              # numpy 1.26.4 override
├── sweep_train.py                       # Parameterized training script (CLI args)
├── launch_sweep.sh                      # Launches all experiments as SLURM jobs
├── package_model.py                     # Package any run: python package_model.py --run NAME
├── package.slurm                        # SLURM wrapper: sbatch package.slurm RUN_NAME
├── convert_fulldata.py                  # Full dataset converter (no val holdout)
├── convert_dataset.py                   # Original 90/10 split converter
├── train_model.py                       # Original single-model training script
└── package_submission.py                # Original packaging script
```

## Phase 1 Results (COMPLETE)

### Training Config
- Model: YOLOv8l (43.9M params)
- Image size: 1280px
- Batch size: 9 (auto-selected for GH200 95GB)
- Optimizer: AdamW (lr=0.001, lrf=0.01, warmup=5 epochs)
- Augmentation: mosaic=1.0, mixup=0.1, copy_paste=0.2, degrees=5, scale=0.5, fliplr=0.5
- Early stopped at epoch 79 (best at epoch 64, patience=15)
- Training time: ~8 minutes on GH200

### Validation Metrics (best epoch)
| Metric | Value |
|--------|-------|
| Precision | 0.814 |
| Recall | 0.800 |
| mAP50 | 0.847 |
| mAP50-95 | 0.592 |

### Competition Score
- **mAP: 0.7615**

---

## Phase 2: Experiment Sweep (IN PROGRESS)

### Sweep Design
All experiments use:
- **Full dataset** (248 train, 10 val for metrics only)
- **150 epochs**, patience=30
- **Inference: conf=0.01, NMS IoU=0.35** (was 0.15/0.5 — should boost recall on dense shelves)

| Experiment | Model | ImgSz | Key Change | SLURM Job |
|---|---|---|---|---|
| `v20_full_l_1280` | YOLOv8l | 1280 | Control: full data + 150ep | 282911 |
| `v21_full_l_1600` | YOLOv8l | 1600 | Higher resolution | 282912 |
| `v22_full_x_1280` | YOLOv8x | 1280 | Bigger model (68M params) | 282913 |
| `v23_full_x_1600` | YOLOv8x | 1600 | Bigger model + higher res | 282914 |
| `v24_full_l_highbox` | YOLOv8l | 1280 | box=12, cls=0.3 (prioritize detection) | 282915 |
| `v25_full_l_moreaug` | YOLOv8l | 1280 | copy_paste=0.4 | 282916 |

### Sweep Results
*(fill in after jobs complete)*

| Experiment | Best Epoch | mAP50 | mAP50-95 | Competition Score | Notes |
|---|---|---|---|---|---|
| `v20_full_l_1280` | | | | | |
| `v21_full_l_1600` | | | | | |
| `v22_full_x_1280` | | | | | |
| `v23_full_x_1600` | | | | | |
| `v24_full_l_highbox` | | | | | |
| `v25_full_l_moreaug` | | | | | |

### Packaging a Model
```bash
# Package best experiment as submission
sbatch package.slurm v22_full_x_1280

# Download to local
scp olivia:/cluster/home/ksv023/NM_AI_2026/task1/submission_v22_full_x_1280/submission.zip ~/Desktop/submission.zip
```

---

## Phase 3: Future Improvements (if time permits)

### TTA (Test-Time Augmentation)
- Multi-scale inference (1024, 1280, 1536) + horizontal flip
- Merge with NMS or weighted boxes fusion
- Must stay within 300s sandbox timeout

### Classification Boost (30% of score)
- Two-stage: detect then classify crops with embedding similarity
- Use `NM_NGD_product_images/` as reference database
- Class weighting / oversampling for rare categories

### Other Ideas
- RT-DETR (transformer detector, available in ultralytics)
- Ensemble multiple models at inference
- Synthetic data from product reference images

## Useful Commands
```bash
# Launch all sweep experiments
bash launch_sweep.sh

# Monitor running jobs
squeue -u $USER
tail -f logs/v2*.log

# Package a specific run
sbatch package.slurm v22_full_x_1280

# Download submission
scp olivia:/cluster/home/ksv023/NM_AI_2026/task1/submission_v22_full_x_1280/submission.zip ~/Desktop/submission.zip

# Check GPU hours
cost
```
