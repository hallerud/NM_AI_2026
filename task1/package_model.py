#!/usr/bin/env python3
"""
Package a specific trained model into submission.zip.
Usage: python package_model.py --run v22_full_x_1280
       python package_model.py --weights runs/v22_full_x_1280/weights/best.pt
"""

import argparse
import shutil
import zipfile
from pathlib import Path

WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")


def make_onnx_run_py(imgsz=1280):
    """Create run.py with optimized inference settings."""
    return f'''import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort


def letterbox(img, new_shape={imgsz}):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * r), int(w * r)
    img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    dh, dw = (new_shape - new_h) // 2, (new_shape - new_w) // 2
    canvas[dh:dh+new_h, dw:dw+new_w] = img_resized
    return canvas, r, dw, dh


def nms(boxes, scores, iou_thresh=0.35):
    """Class-agnostic NMS."""
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou_val = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou_val < iou_thresh]
    return keep


def postprocess(outputs, conf_thresh=0.01, iou_thresh=0.35, r=1.0, dw=0, dh=0):
    preds = outputs[0][0].T  # (num_preds, 4+nc)
    boxes_raw, scores_all = preds[:, :4], preds[:, 4:]
    max_scores = scores_all.max(axis=1)
    class_ids = scores_all.argmax(axis=1)

    mask = max_scores > conf_thresh
    boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]
    if len(boxes_raw) == 0:
        return [], [], []

    # Convert cx,cy,w,h to x1,y1,x2,y2 then scale back
    x1 = (boxes_raw[:, 0] - boxes_raw[:, 2] / 2 - dw) / r
    y1 = (boxes_raw[:, 1] - boxes_raw[:, 3] / 2 - dh) / r
    w = boxes_raw[:, 2] / r
    h = boxes_raw[:, 3] / r

    # xywh format for output
    boxes_out = np.stack([x1, y1, w, h], axis=1)

    # Per-class NMS
    keep_all = []
    for cid in np.unique(class_ids):
        m = class_ids == cid
        idx = np.where(m)[0]
        cls_keep = nms(boxes_out[idx], max_scores[idx], iou_thresh)
        keep_all.extend(idx[cls_keep])

    keep_all = np.array(keep_all)
    return boxes_out[keep_all], max_scores[keep_all], class_ids[keep_all]


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
    print(f"Processing {{len(image_files)}} images...")

    for i, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])
        img = np.array(Image.open(img_path).convert("RGB"))
        processed, r, dw, dh = letterbox(img, {imgsz})
        blob = np.transpose(processed.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
        outputs = session.run(None, {{input_name: blob}})
        boxes, scores, cids = postprocess(outputs, r=r, dw=dw, dh=dh)

        for box, score, cid in zip(boxes, scores, cids):
            predictions.append({{
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": [round(float(x), 1) for x in box],
                "score": round(float(score), 4),
            }})

        if (i + 1) % 10 == 0:
            print(f"  {{i+1}}/{{len(image_files)}} done")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Done! {{len(predictions)}} predictions")


if __name__ == "__main__":
    main()
'''


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Run name, e.g. v22_full_x_1280")
    group.add_argument("--weights", help="Direct path to best.pt")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size")
    parser.add_argument("--out", default=None, help="Output dir name")
    args = parser.parse_args()

    if args.run:
        weights = WORKDIR / "runs" / args.run / "weights" / "best.pt"
        out_name = args.out or f"submission_{args.run}"
    else:
        weights = Path(args.weights)
        out_name = args.out or "submission_custom"

    if not weights.exists():
        print(f"ERROR: {weights} not found")
        return

    # Detect imgsz from run args if not specified
    imgsz = args.imgsz
    if imgsz is None:
        # Try to guess from run name
        name = weights.parent.parent.name
        if "1600" in name:
            imgsz = 1600
        elif "1920" in name:
            imgsz = 1920
        else:
            imgsz = 1280

    output_dir = WORKDIR / out_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging = output_dir / "staging"
    staging.mkdir(parents=True)

    print(f"Packaging: {weights}")
    print(f"  ONNX export at imgsz={imgsz}")

    from ultralytics import YOLO
    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=imgsz, half=True, opset=17, simplify=True)

    onnx_path = weights.with_suffix(".onnx")
    shutil.copy2(onnx_path, staging / "best.onnx")

    with open(staging / "run.py", "w") as f:
        f.write(make_onnx_run_py(imgsz))

    # Create zip
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(staging.iterdir()):
            zf.write(fpath, fpath.name)

    # Print summary
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:30s} {info.file_size / 1024 / 1024:8.1f} MB")

    total = zip_path.stat().st_size / 1024 / 1024
    print(f"\n  Total: {total:.1f} MB {'OK' if total <= 420 else 'OVER LIMIT!'}")
    print(f"  Output: {zip_path}")
    print(f"\n  scp olivia:{zip_path} ~/Desktop/submission.zip")


if __name__ == "__main__":
    main()
