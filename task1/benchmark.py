#!/usr/bin/env python3
"""
Benchmark models on training images using COCO mAP evaluation.
Scores are inflated (model trained on these images) but relative ranking is reliable.

Mirrors the contest scoring: 70% detection mAP + 30% classification mAP.

Usage: python benchmark.py --runs v63_x_full v64_l_full
       python benchmark.py --runs v63_x_full v64_l_full --ensemble
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")
ANNOTATIONS = WORKDIR / "data" / "train" / "annotations.json"
IMAGES_DIR = WORKDIR / "data" / "train" / "images"


def letterbox(img_np, target_size):
    h, w = img_np.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dh, dw = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, scale, dw, dh


def decode_yolo_onnx(outputs, scale, dw, dh, img_w, img_h, conf_thresh=0.001):
    preds = outputs[0][0].T
    boxes_raw, scores_all = preds[:, :4], preds[:, 4:]
    max_scores = scores_all.max(axis=1)
    class_ids = scores_all.argmax(axis=1)

    mask = max_scores > conf_thresh
    boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]
    if len(boxes_raw) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([])

    x1 = np.clip((boxes_raw[:, 0] - boxes_raw[:, 2] / 2 - dw) / scale, 0, img_w)
    y1 = np.clip((boxes_raw[:, 1] - boxes_raw[:, 3] / 2 - dh) / scale, 0, img_h)
    x2 = np.clip((boxes_raw[:, 0] + boxes_raw[:, 2] / 2 - dw) / scale, 0, img_w)
    y2 = np.clip((boxes_raw[:, 1] + boxes_raw[:, 3] / 2 - dh) / scale, 0, img_h)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    return boxes_xywh, max_scores, class_ids


def run_ultralytics(model, img_path, imgsz, conf_thresh=0.001):
    """Run inference using ultralytics .pt model (no ONNX export needed)."""
    results = model(
        str(img_path), device=0, imgsz=imgsz,
        conf=conf_thresh, iou=0.6, max_det=300, verbose=False,
    )
    boxes, scores, cids = [], [], []
    for r in results:
        if r.boxes is None:
            continue
        for j in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(r.boxes.conf[j].item()))
            cids.append(int(r.boxes.cls[j].item()))
    return np.array(boxes) if boxes else np.zeros((0, 4)), np.array(scores), np.array(cids)


def evaluate_coco(predictions, ann_file):
    """Compute contest-style score: 70% detection + 30% classification."""
    coco_gt = COCO(str(ann_file))

    if not predictions:
        print("  No predictions!")
        return 0.0

    # Classification mAP (normal COCO eval)
    coco_dt = coco_gt.loadRes(predictions)
    eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
    eval_cls.evaluate()
    eval_cls.accumulate()
    print("\n  Classification mAP (correct category required):")
    eval_cls.summarize()
    cls_map = eval_cls.stats[0]

    # Detection mAP (ignore category — set all to same id)
    preds_det = [dict(p, category_id=0) for p in predictions]
    anns_det = json.loads(json.dumps(coco_gt.dataset))
    for ann in anns_det["annotations"]:
        ann["category_id"] = 0
    anns_det["categories"] = [{"id": 0, "name": "product", "supercategory": "product"}]

    # Write temp files for detection eval
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(anns_det, f)
        det_gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds_det, f)
        det_dt_path = f.name

    coco_gt_det = COCO(det_gt_path)
    coco_dt_det = coco_gt_det.loadRes(det_dt_path)
    eval_det = COCOeval(coco_gt_det, coco_dt_det, "bbox")
    eval_det.evaluate()
    eval_det.accumulate()
    print("\n  Detection mAP (category ignored):")
    eval_det.summarize()
    det_map = eval_det.stats[0]

    Path(det_gt_path).unlink()
    Path(det_dt_path).unlink()

    contest_score = 0.7 * det_map + 0.3 * cls_map
    return contest_score, det_map, cls_map


def benchmark_single(run_name, imgsz=None):
    """Benchmark a single model using ultralytics .pt weights."""
    from ultralytics import YOLO

    weights = WORKDIR / "runs" / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"  {weights} not found")
        return

    if imgsz is None:
        # Read imgsz from training args.yaml
        args_yaml = WORKDIR / "runs" / run_name / "args.yaml"
        if args_yaml.exists():
            import yaml
            with open(args_yaml) as f:
                train_args = yaml.safe_load(f)
            imgsz = train_args.get("imgsz", 1280)
        else:
            imgsz = 1600 if "1600" in run_name else 1280

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {run_name} (imgsz={imgsz})")
    print(f"{'='*60}")

    model = YOLO(str(weights))
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"  Images: {len(image_files)}")

    predictions = []
    t0 = time.time()
    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        boxes, scores, cids = run_ultralytics(model, img_path, imgsz)
        for box, score, cid in zip(boxes, scores, cids):
            predictions.append({
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": [round(float(x), 1) for x in box],
                "score": round(float(score), 4),
            })
    elapsed = time.time() - t0
    print(f"  Inference: {elapsed:.1f}s ({elapsed/len(image_files):.2f}s/img)")
    print(f"  Predictions: {len(predictions)}")

    score, det_map, cls_map = evaluate_coco(predictions, ANNOTATIONS)
    print(f"\n  Contest score: {score:.4f} (70%*{det_map:.4f} det + 30%*{cls_map:.4f} cls)")
    return score


def benchmark_ensemble(run_names, imgsz_list=None):
    """Benchmark ensemble using WBF."""
    from ultralytics import YOLO
    from ensemble_boxes import weighted_boxes_fusion

    models = []
    for i, run_name in enumerate(run_names):
        weights = WORKDIR / "runs" / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"  {weights} not found")
            return
        if imgsz_list and i < len(imgsz_list):
            imgsz = imgsz_list[i]
        else:
            args_yaml = WORKDIR / "runs" / run_name / "args.yaml"
            if args_yaml.exists():
                import yaml
                with open(args_yaml) as f:
                    train_args = yaml.safe_load(f)
                imgsz = train_args.get("imgsz", 1280)
            else:
                imgsz = 1600 if "1600" in run_name else 1280
        print(f"  Loading {run_name} (imgsz={imgsz})")
        models.append((YOLO(str(weights)), imgsz))

    print(f"\n{'='*60}")
    print(f"  Benchmarking ensemble: {run_names}")
    print(f"{'='*60}")

    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"  Images: {len(image_files)}")

    predictions = []
    t0 = time.time()
    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        all_boxes, all_scores, all_labels = [], [], []
        for model, imgsz in models:
            boxes, scores, cids = run_ultralytics(model, img_path, imgsz)
            if len(boxes) > 0:
                # Normalize to [0,1] for WBF
                norm_boxes = np.zeros((len(boxes), 4))
                norm_boxes[:, 0] = np.clip(boxes[:, 0] / img_w, 0, 1)
                norm_boxes[:, 1] = np.clip(boxes[:, 1] / img_h, 0, 1)
                norm_boxes[:, 2] = np.clip((boxes[:, 0] + boxes[:, 2]) / img_w, 0, 1)
                norm_boxes[:, 3] = np.clip((boxes[:, 1] + boxes[:, 3]) / img_h, 0, 1)
                all_boxes.append(norm_boxes.tolist())
                all_scores.append(scores.tolist())
                all_labels.append(cids.tolist())
            else:
                all_boxes.append([])
                all_scores.append([])
                all_labels.append([])

        if all(len(b) == 0 for b in all_boxes):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=[1.0] * len(models), iou_thr=0.6, skip_box_thr=0.001,
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1 = box[0] * img_w
            y1 = box[1] * img_h
            w = (box[2] - box[0]) * img_w
            h = (box[3] - box[1]) * img_h
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(score), 4),
            })
    elapsed = time.time() - t0
    print(f"  Inference: {elapsed:.1f}s ({elapsed/len(image_files):.2f}s/img)")
    print(f"  Predictions: {len(predictions)}")

    score, det_map, cls_map = evaluate_coco(predictions, ANNOTATIONS)
    print(f"\n  Contest score: {score:.4f} (70%*{det_map:.4f} det + 30%*{cls_map:.4f} cls)")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--ensemble", action="store_true", help="Also benchmark as ensemble")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Model Benchmark (contest-style scoring)")
    print("  NOTE: Scores inflated (trained on eval images)")
    print("  Use for RELATIVE comparison only")
    print("=" * 60)

    scores = {}
    for run_name in args.runs:
        scores[run_name] = benchmark_single(run_name)

    if args.ensemble and len(args.runs) > 1:
        scores["ENSEMBLE"] = benchmark_ensemble(args.runs, args.imgsz)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for name, score in sorted(scores.items(), key=lambda x: x[1] or 0, reverse=True):
        print(f"  {name:30s} {score:.4f}")


if __name__ == "__main__":
    main()
