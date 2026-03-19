#!/usr/bin/env python3
"""
Package trained model into submission.zip.
Run inside the container via step3c_package.slurm

Automatically finds the latest trained weights, decides whether
to use .pt or ONNX format, creates run.py, and zips everything.
"""

import shutil
import zipfile
from pathlib import Path

# ================================================================
WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")
OUTPUT_DIR = WORKDIR / "trained_submission"
# ================================================================


def find_best_weights():
    """Find the most recent best.pt from training runs."""
    runs_dir = WORKDIR / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory at {runs_dir}. Train first!")

    # Find all best.pt files, pick most recent
    bests = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not bests:
        raise FileNotFoundError("No best.pt found in runs/. Training may not be done yet.")

    latest = bests[-1]
    print(f"  Found: {latest}")
    print(f"  Size: {latest.stat().st_size / 1024 / 1024:.1f} MB")
    return latest


def make_pt_run_py():
    """Create run.py that uses ultralytics .pt weights."""
    return '''import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("best.pt")
    predictions = []

    image_files = [f for f in sorted(Path(args.input).iterdir())
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])

        results = model(
            str(img_path),
            device=device,
            imgsz=1280,
            conf=0.15,
            iou=0.5,
            max_det=300,
            verbose=False,
        )

        for r in results:
            if r.boxes is None:
                continue
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[j].item()),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[j].item()), 3),
                })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(image_files)} images done")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Done! {len(predictions)} predictions")


if __name__ == "__main__":
    main()
'''


def make_onnx_run_py():
    """Create run.py that uses ONNX Runtime."""
    return '''import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort


def letterbox(img, new_shape=1280):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * r), int(w * r)
    img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    dh, dw = (new_shape - new_h) // 2, (new_shape - new_w) // 2
    canvas[dh:dh+new_h, dw:dw+new_w] = img_resized
    return canvas, r, dw, dh


def postprocess(outputs, conf_thresh=0.15, iou_thresh=0.5, r=1.0, dw=0, dh=0):
    preds = outputs[0][0].T
    boxes, scores = preds[:, :4], preds[:, 4:]
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)
    mask = max_scores > conf_thresh
    boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]
    if len(boxes) == 0:
        return [], [], []
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    keep = []
    for cid in np.unique(class_ids):
        m = class_ids == cid
        idx = np.where(m)[0]
        s, cx1, cy1, cx2, cy2 = max_scores[m], x1[m], y1[m], x2[m], y2[m]
        order = s.argsort()[::-1]
        ck = []
        while len(order) > 0:
            i = order[0]; ck.append(idx[i])
            if len(order) == 1: break
            xx1 = np.maximum(cx1[i], cx1[order[1:]])
            yy1 = np.maximum(cy1[i], cy1[order[1:]])
            xx2 = np.minimum(cx2[i], cx2[order[1:]])
            yy2 = np.minimum(cy2[i], cy2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            a1 = (cx2[i]-cx1[i]) * (cy2[i]-cy1[i])
            a2 = (cx2[order[1:]]-cx1[order[1:]]) * (cy2[order[1:]]-cy1[order[1:]])
            iou_val = inter / (a1 + a2 - inter + 1e-6)
            order = order[1:][iou_val < iou_thresh]
        keep.extend(ck)
    keep = np.array(keep)
    fx1, fy1 = (x1[keep] - dw) / r, (y1[keep] - dh) / r
    fx2, fy2 = (x2[keep] - dw) / r, (y2[keep] - dh) / r
    return np.stack([fx1, fy1, fx2-fx1, fy2-fy1], axis=1), max_scores[keep], class_ids[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    session = ort.InferenceSession("best.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions = []
    image_files = [f for f in sorted(Path(args.input).iterdir())
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"Processing {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])
        img = np.array(Image.open(img_path).convert("RGB"))
        processed, r, dw, dh = letterbox(img, 1280)
        blob = np.transpose(processed.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
        outputs = session.run(None, {input_name: blob})
        boxes, scores, cids = postprocess(outputs, r=r, dw=dw, dh=dh)
        for box, score, cid in zip(boxes, scores, cids):
            predictions.append({
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": [round(float(x), 1) for x in box],
                "score": round(float(score), 3),
            })
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(image_files)} done")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Done! {len(predictions)} predictions")


if __name__ == "__main__":
    main()
'''


def main():
    print("=" * 50)
    print("  Packaging Trained Submission")
    print("=" * 50)

    # Find weights
    print("\n1. Finding trained weights...")
    weights = find_best_weights()
    pt_size = weights.stat().st_size / 1024 / 1024

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    staging = OUTPUT_DIR / "staging"
    staging.mkdir(parents=True)

    # Always export to ONNX for sandbox compatibility
    # (sandbox has ultralytics==8.1.0, we trained with 8.4.24)
    print(f"\n2. Exporting to ONNX FP16 ({pt_size:.1f} MB .pt)")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=1280, half=True, opset=17, simplify=True)
    onnx_path = weights.with_suffix(".onnx")
    shutil.copy2(onnx_path, staging / "best.onnx")
    onnx_size = (staging / "best.onnx").stat().st_size / 1024 / 1024
    print(f"   ONNX size: {onnx_size:.1f} MB")
    with open(staging / "run.py", "w") as f:
        f.write(make_onnx_run_py())

    print("   Created run.py")

    # Zip
    print("\n3. Creating submission.zip...")
    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.iterdir()):
            zf.write(f, f.name)

    # Verify
    print("\n4. Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"   {info.filename:30s} {info.file_size / 1024 / 1024:8.1f} MB")
        assert "run.py" in zf.namelist(), "run.py not at zip root!"

    total = zip_path.stat().st_size / 1024 / 1024
    print(f"\n   Total: {total:.1f} MB {'✓' if total <= 420 else '⚠ OVER LIMIT'}")

    print(f"\n{'=' * 50}")
    print(f"  DONE! Upload this file:")
    print(f"  {zip_path}")
    print(f"\n  Download to Mac:")
    print(f"  scp ksv023@login.nris.no:{zip_path} ~/Desktop/submission.zip")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
