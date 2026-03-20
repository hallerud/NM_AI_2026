# NorgesGruppen Object Detection Competition - Task 1

## Current Status
- **Best Score: mAP 0.9154** (3-model ensemble + TTA hflip) — up from 0.9137
- **Deadline: ~2026-03-21**
- **Submissions: 3 remaining today (2026-03-20)**
- **GPU hours remaining: ~860 of 1000** (expires 2026-04-01)

## Submissions History

| # | Model | Export | Conf/IoU | Score | Notes |
|---|---|---|---|---|---|
| 1 | yolov8l_v13 | ONNX FP16, 1280 | 0.15/0.5 | 0.7615 | Baseline |
| 2 | v32_x_moreaug_16002 | ONNX FP16, 1600 | 0.15/0.5 | 0.7716 | YOLOv8x, more aug |
| 3 | v41_x_long200 | ONNX FP32, 1600 | 0.001/0.6 | 0.7773 | 200ep, FP32, low conf |
| 4 | Ensemble: v63+v41+v64 | 3x ONNX FP16 | 0.001/0.6 WBF | 0.9137 | 3-model WBF ensemble |
| **5** | **Ensemble + TTA hflip** | **3x ONNX FP16** | **0.001/0.6 WBF** | **0.9154** | **+TTA: orig+hflip per model, 6 pred sets** |

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

### 5. Test-Time Augmentation — horizontal flip (+0.0017 mAP)
Each model runs on original + horizontally flipped image, doubling prediction sets (3→6) for WBF.
Small but consistent improvement: 0.9137 → 0.9154 on contest.

## What Didn't Work
- **Product reference images as training data**: No improvement (v50 = v53)
- **RT-DETR**: Classification head doesn't converge with 356 classes on 248 images
- **Multi-scale training**: No measurable improvement
- **Heavy augmentation**: No measurable improvement over baseline aug

## What To Try Next (Priority Order)

### DONE: Test-Time Augmentation (hflip)
Implemented and benchmarked. Horizontal flip TTA adds +0.0036 local score (0.7049 → 0.7086).
Now baked into ensemble_run.py — each model runs orig + hflip = 6 prediction sets for WBF.
WBF IoU sweep (0.5–0.65) showed minimal difference; iou=0.6 remains optimal.

### DONE: WBF parameter sweep
Tested iou_thr={0.5, 0.55, 0.6, 0.65} — all within 0.0002 of each other. Default 0.6 is fine.

## Roadmap to 0.95+ mAP

Current score breakdown (estimated): ~0.95 det mAP × 0.70 + ~0.88 cls mAP × 0.30 ≈ 0.915.
Detection is already strong. The biggest gains are in **classification accuracy**.
To reach 0.95: need e.g. 0.96 det × 0.70 + 0.93 cls × 0.30 = 0.672 + 0.279 = 0.951.

### 1. Classification re-ranking with embeddings (HIGH — targets 30% of score)
The single most promising path to 0.95+. The idea:
1. Use current ensemble for detection (already strong)
2. Crop each detected bounding box from the image
3. Extract embeddings using a pretrained model (timm has EfficientNet/ConvNeXt, pre-installed in sandbox)
4. Compare each crop embedding to pre-computed embeddings of the 326 product reference images
5. Re-assign category_id based on nearest-neighbor similarity

**Why this could work big:**
- 30% of the score is classification. If det mAP is ~0.95 but cls mAP is ~0.88, fixing misclassifications could push total to 0.95+
- Product reference images exist: 326 products × 4 views (front/back/left/right) in `data/NM_NGD_product_images/`
- timm 0.9.12 is pre-installed in the sandbox — no extra weight files needed for embeddings
- Can be done in run.py post-processing — no retraining required

**Implementation plan:**
- Pre-compute reference embeddings offline, save as .npy (~1MB)
- In run.py: crop detections → embed → cosine similarity → re-rank class IDs
- Ship: 3 ONNX models + 1 .npy file (well under 420MB)
- Must fit within 300s timeout

### 2. Multi-scale TTA (MEDIUM — +0.5-1%)
Add a second scale (e.g. 0.8x) in addition to hflip.
3 models × 2 scales × 2 flips = 12 prediction sets. More diversity for WBF.
Risk: must stay under 300s sandbox timeout.

### 3. Train a 4th complementary model (MEDIUM)
Current: 2× YOLOv8x + 1× YOLOv8l. A small YOLOv8s (~20MB ONNX) could add diversity.
73MB headroom remains in the 420MB weight budget.
Different augmentation or image size would maximize ensemble diversity.

## Available Data

### 1. Shelf images + annotations (`data/train/`)
- **248 images** annotated in `annotations.json` (COCO format), but only **210 on disk** — the other 38 are likely the hidden test set
- **22,731 annotations** across **356 product categories**
- Images vary widely: 481×399 to 5712×4624 pixels
- Dense scenes: avg **92 annotations/image**, max 235
- **Long-tail distribution**: median 28 annotations per category, but 110 categories have <10 examples and 74 have <5
- Annotation format: `[x, y, width, height]` (COCO bbox), with `category_id`, `image_id`, `area`, `iscrowd`
- Category names are product descriptions, e.g. "FRØKRISP KNEKKEBRØD ØKOLOGISK 170G BERIT"

### 2. Product reference images (`data/NM_NGD_product_images/`)
- **345 product folders** (named by EAN/barcode), each containing up to 7 views: front, back, left, right, top, bottom, main
- Covers 345 of 356 categories — **11 categories have no product images**
- **Not currently used** in training or inference
- No explicit mapping file from folder name (EAN) to category_id — mapping must be inferred or built
- These are clean studio shots of individual products, very different from the shelf images
- **Potential uses**: classification re-ranking (embed crops → match to reference), few-shot classification, data augmentation

### 3. What's NOT available
- No validation set — must use contest submissions or cross-validation on training data
- No test images — test set is hidden, evaluation is server-side
- Category metadata is minimal (name + supercategory "product") — no EAN-to-category mapping provided

## Scoring Formula
- **70% detection mAP** — did you find the products? (IoU >= 0.5, category ignored)
- **30% classification mAP** — correct product ID? (IoU >= 0.5 AND correct category_id)

## Local Benchmark
benchmark.py computes contest-style scores on training images (inflated but reliable for relative comparison).
Benchmark offset: local score + ~0.08 ≈ contest score (based on v41: 0.693 local → 0.777 contest).

| Model | Det mAP | Cls mAP | Local Score |
|---|---|---|---|
| **ENSEMBLE + TTA hflip** | **0.6795** | **0.7766** | **0.7086** |
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
6. **TTA hflip helps modestly** — +0.0036 local score, consistent across WBF IoU settings
7. **WBF IoU threshold is robust** — 0.5–0.65 all perform within 0.0002

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

## Key Files
| File | Purpose |
|------|---------|
| `ensemble_run.py` | Submission run.py — ONNX inference + TTA + WBF. This is what runs in the sandbox |
| `benchmark.py` | Local evaluation (contest-style scoring). Supports `--tta`, `--wbf-iou`, `--wbf-weights` |
| `package_ensemble.py` | Exports .pt → ONNX and zips submission (run.py + models) |
| `package_ensemble.slurm` | Slurm wrapper for packaging |
| `benchmark_parallel.slurm` | Slurm wrapper for single benchmark config (for parallel sweeps) |
| `make_dataset.py` | Builds YOLO-format datasets from COCO annotations |
| `data/train/annotations.json` | COCO-format ground truth (210 shelf images, 356 categories) |
| `data/train/images/` | Training shelf images |
| `data/NM_NGD_product_images/` | Product reference images (326 products × 4 views) — unused so far, key for re-ranking |
| `runs/` | Training run outputs (weights, args.yaml, etc.) |

## Useful Commands
```bash
# Benchmark models locally (contest-style scoring)
sbatch benchmark.slurm "v63_x_full v64_l_full v41_x_long200"

# Benchmark with TTA + custom WBF params
sbatch benchmark_parallel.slurm "--runs v63_x_full v41_x_long200 v64_l_full --ensemble --tta --wbf-iou 0.6"

# Package ensemble submission
sbatch package_ensemble.slurm "--runs v63_x_full v41_x_long200 v64_l_full --half 1 1 1 --out submission_tta"

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
