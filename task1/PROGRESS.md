# NorgesGruppen Object Detection Competition - Task 1

## Current Status
- **Best Score: mAP 0.9137** (3-model ensemble, rank TBD) — previous best 0.7773
- **Deadline: ~2026-03-21**
- **Submissions: 3 per day**
- **GPU hours remaining: ~860 of 1000** (expires 2026-04-01)

## Submissions History

| # | Model | Export | Conf/IoU | Score | Notes |
|---|---|---|---|---|---|
| 1 | yolov8l_v13 | ONNX FP16, 1280 | 0.15/0.5 | 0.7615 | Baseline |
| 2 | v32_x_moreaug_16002 | ONNX FP16, 1600 | 0.15/0.5 | 0.7716 | YOLOv8x, more aug |
| 3 | v41_x_long200 | ONNX FP32, 1600 | 0.001/0.6 | 0.7773 | 200ep, FP32, low conf |
| **4** | **Ensemble: v63+v41+v64** | **3x ONNX FP16** | **0.001/0.6 WBF** | **0.9137** | **3-model WBF ensemble** |

## What Worked (Key Breakthroughs)

### 1. Ensemble with Weighted Boxes Fusion (+0.14 mAP)
The single biggest improvement. Three diverse models merged with WBF:
- **v63_x_full**: YOLOv8x, imgsz=1600, 200ep, trained on yolo_full (248 images)
- **v41_x_long200**: YOLOv8x, imgsz=1600, 200ep, older config (yolo_dataset_full)
- **v64_l_full**: YOLOv8l, imgsz=1280, 200ep, trained on yolo_full (248 images)
- Diversity from: different model sizes (x vs l), different image sizes (1600 vs 1280), different training data configs
- `ensemble-boxes` is pre-installed in the sandbox
- Submission allows up to 3 weight files, 420MB total

### 2. Low confidence threshold (conf=0.001)
COCO mAP evaluation rewards recall at all confidence levels. Low threshold + WBF lets the metric find the optimal operating point.

### 3. Higher IoU for NMS (iou=0.6)
Dense shelf scenes have many overlapping products. Higher IoU threshold keeps more valid detections.

### 4. FP16 for all 3 models
Fits 3 models under 420MB limit (131+131+84 = 347MB). FP16 quantization loss is compensated by ensemble diversity.

## What Didn't Work
- **Product reference images as training data**: No improvement (v50 = v53)
- **RT-DETR**: Classification head doesn't converge with 356 classes on 248 images
- **Multi-scale training**: No measurable improvement
- **Heavy augmentation**: No measurable improvement over baseline aug

## What To Try Next (Priority Order)

### 1. Test-Time Augmentation (HIGH — likely +1-3%)
Run each model at multiple scales (e.g. original + flipped) and merge all predictions with WBF.
Current: 3 models x 1 scale = 3 inference passes.
With TTA: 3 models x 2 scales x 2 flips = 12 passes. Must stay under 300s timeout.
Easy to implement — just add flip/scale loops in ensemble_run.py.

### 2. Tune WBF parameters (MEDIUM — likely +0.5-1%)
Current: iou_thr=0.6, skip_box_thr=0.001, weights=[1,1,1].
Try: different iou thresholds (0.5, 0.55, 0.65), different model weights, skip thresholds.
Use benchmark.py to test locally before submitting.

### 3. Add a third architecture (MEDIUM — if it fits)
Current ensemble is 2x YOLOv8x + 1x YOLOv8l — all YOLO. A different architecture would add more diversity.
Options: Faster R-CNN (torchvision), FCOS, or a custom model with timm backbone.
Challenge: must fit within 420MB total and 300s timeout.

### 4. Classification re-ranking (MEDIUM — targets 30% of score)
Detect with ensemble, crop each detection, compare crops to product reference images using embedding similarity (timm backbone). Could fix misclassifications.
Product images: 326 products with front/back/left/right views in data/NM_NGD_product_images/.

### 5. Larger image size for one model (LOW)
Train one model at imgsz=1920 or 2048 for better small object detection.
Risk: ONNX model gets huge, inference slow.

### 6. More training epochs / better schedule (LOW)
Most models plateau by epoch 150-200. Diminishing returns.

## Scoring Formula
- **70% detection mAP** — did you find the products? (IoU >= 0.5, category ignored)
- **30% classification mAP** — correct product ID? (IoU >= 0.5 AND correct category_id)

## Local Benchmark
benchmark.py computes contest-style scores on training images (inflated but reliable for relative comparison).
Benchmark offset: local score + ~0.08 ≈ contest score (based on v41: 0.693 local → 0.777 contest).

| Model | Det mAP | Cls mAP | Local Score |
|---|---|---|---|
| ENSEMBLE (3 models) | 0.6761 | 0.7720 | 0.7049 |
| v41_x_long200 | 0.6649 | 0.7587 | 0.6930 |
| v63_x_full | 0.6635 | 0.7588 | 0.6921 |
| v64_l_full | 0.6493 | 0.7274 | 0.6727 |

## Key Lessons
1. **Ensemble > single model** — 3 diverse models with WBF was worth +0.14 mAP
2. **Validation on training data is useless** — must use held-out val or submit to contest
3. **Inference settings matter** — conf=0.001, iou=0.6 much better than conf=0.15, iou=0.5
4. **FP16 is fine when ensembling** — quantization loss per model is compensated by ensemble averaging
5. **Detection is the bottleneck** — 70% of score, and our det mAP is lower than cls mAP

## Infrastructure

### Cluster (Olivia)
- **GPU nodes:** ARM64 (Neoverse V2) + GH200 (96 GB HBM3)
- **Container:** `/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif`
- **Account:** nn11127k (~860 GPU hours remaining, expires 2026-04-01)

### Sandbox Constraints
| Resource | Limit |
|----------|-------|
| GPU | NVIDIA L4 (24 GB VRAM) |
| CPU | 4 vCPU |
| Memory | 8 GB |
| Timeout | 300 seconds |
| Max weight files | 3, 420 MB total |
| Pre-installed | ensemble-boxes 1.0.9, ultralytics 8.1.0, timm 0.9.12, onnxruntime-gpu 1.20.0 |

## Useful Commands
```bash
# Benchmark models locally (contest-style scoring)
sbatch benchmark.slurm "v63_x_full v64_l_full v41_x_long200"

# Package ensemble submission
sbatch package_ensemble.slurm "v63_x_full v41_x_long200 v64_l_full"

# Package single model
sbatch package.slurm RUN_NAME

# Build datasets
python3 make_dataset.py                  # shelf images only, proper split
python3 make_dataset.py --product_images # + product reference images

# Monitor
squeue -u $USER
python3 compare_runs.py
cost -p nn11127k

# Download
scp olivia:/cluster/home/ksv023/NM_AI_2026/task1/submission_ensemble/submission.zip ~/Desktop/
```
