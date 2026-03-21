"""
Ensemble inference: YOLOv8x + RT-DETR merged with Weighted Boxes Fusion.
Runs both models on each image, combines predictions via WBF.

Usage: python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


# --- Preprocessing ---

def letterbox(img_np, target_size):
    """Resize with padding to square, preserving aspect ratio."""
    h, w = img_np.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dh, dw = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, scale, dw, dh


def to_blob(img_np):
    """HWC uint8 -> NCHW float32 [0,1]."""
    return np.transpose(img_np.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]


# --- Postprocessing ---

def decode_yolo(outputs, scale, dw, dh, img_w, img_h, conf_thresh=0.001):
    """Decode raw YOLOv8 ONNX output to (boxes_norm, scores, class_ids).
    Returns boxes normalized to [0,1] relative to original image (for WBF).
    """
    preds = outputs[0][0].T  # (num_preds, 4+nc)
    boxes_raw, scores_all = preds[:, :4], preds[:, 4:]
    max_scores = scores_all.max(axis=1)
    class_ids = scores_all.argmax(axis=1)

    mask = max_scores > conf_thresh
    boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]
    if len(boxes_raw) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([])

    # cxcywh (in padded coords) -> x1y1x2y2 (in original image coords)
    x1 = (boxes_raw[:, 0] - boxes_raw[:, 2] / 2 - dw) / scale
    y1 = (boxes_raw[:, 1] - boxes_raw[:, 3] / 2 - dh) / scale
    x2 = (boxes_raw[:, 0] + boxes_raw[:, 2] / 2 - dw) / scale
    y2 = (boxes_raw[:, 1] + boxes_raw[:, 3] / 2 - dh) / scale

    # Clip and normalize to [0,1] for WBF
    x1 = np.clip(x1 / img_w, 0, 1)
    y1 = np.clip(y1 / img_h, 0, 1)
    x2 = np.clip(x2 / img_w, 0, 1)
    y2 = np.clip(y2 / img_h, 0, 1)

    boxes_norm = np.stack([x1, y1, x2, y2], axis=1)
    return boxes_norm, max_scores, class_ids


def run_model(session, img_np, imgsz, img_w, img_h, conf_thresh=0.001):
    """Run one ONNX model and return normalized boxes, scores, class_ids."""
    processed, scale, dw, dh = letterbox(img_np, imgsz)
    blob = to_blob(processed)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return decode_yolo(outputs, scale, dw, dh, img_w, img_h, conf_thresh)


def run_model_tta(session, img_np, imgsz, img_w, img_h, conf_thresh=0.001,
                   scales=(1.0,)):
    """Run one ONNX model with TTA (each scale × {original, hflip}).
    For scales != 1.0, the image is resized before letterboxing to simulate
    running at a different resolution, while keeping the ONNX input size fixed.
    Returns list of (boxes_norm, scores, class_ids) — one per augmentation.
    """
    results = []

    for scale_factor in scales:
        if scale_factor == 1.0:
            img_scaled = img_np
            sw, sh = img_w, img_h
        else:
            # Resize image to simulate smaller effective resolution
            sh = int(img_h * scale_factor)
            sw = int(img_w * scale_factor)
            img_scaled = np.array(Image.fromarray(img_np).resize((sw, sh), Image.BILINEAR))

        # Original at this scale
        boxes, scores, cids = run_model(session, img_scaled, imgsz, sw, sh, conf_thresh)
        results.append((boxes, scores, cids))

        # Horizontal flip at this scale
        img_flip = img_scaled[:, ::-1, :].copy()
        boxes_f, scores_f, cids_f = run_model(session, img_flip, imgsz, sw, sh, conf_thresh)
        if len(boxes_f) > 0:
            # Flip boxes back: x1_new = 1 - x2_old, x2_new = 1 - x1_old
            x1 = 1.0 - boxes_f[:, 2]
            x2 = 1.0 - boxes_f[:, 0]
            boxes_f[:, 0] = x1
            boxes_f[:, 2] = x2
        results.append((boxes_f, scores_f, cids_f))

    return results


def ensemble_predict(models, img_np, img_w, img_h, iou_thresh=0.6, skip_thresh=0.001):
    """Run all models with TTA, merge with Weighted Boxes Fusion.
    The smallest model (by imgsz) gets multi-scale TTA for extra diversity.
    """
    all_boxes, all_scores, all_labels = [], [], []
    weights = []

    # Find the smallest imgsz to give it multi-scale TTA
    min_imgsz = min(imgsz for _, imgsz, _ in models)

    for session, imgsz, weight in models:
        scales = (1.0, 0.85) if imgsz == min_imgsz else (1.0,)
        tta_results = run_model_tta(session, img_np, imgsz, img_w, img_h, skip_thresh, scales=scales)
        for boxes, scores, cids in tta_results:
            if len(boxes) > 0:
                all_boxes.append(boxes.tolist())
                all_scores.append(scores.tolist())
                all_labels.append(cids.tolist())
            else:
                all_boxes.append([])
                all_scores.append([])
                all_labels.append([])
            weights.append(weight)

    if all(len(b) == 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([])

    boxes_out, scores_out, labels_out = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights,
        iou_thr=iou_thresh,
        skip_box_thr=skip_thresh,
    )

    # WBF returns x1y1x2y2 normalized — convert to x1y1wh in pixel coords
    boxes_pixel = np.zeros_like(boxes_out)
    boxes_pixel[:, 0] = boxes_out[:, 0] * img_w
    boxes_pixel[:, 1] = boxes_out[:, 1] * img_h
    boxes_pixel[:, 2] = (boxes_out[:, 2] - boxes_out[:, 0]) * img_w
    boxes_pixel[:, 3] = (boxes_out[:, 3] - boxes_out[:, 1]) * img_h

    return boxes_pixel, scores_out, labels_out


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    script_dir = Path(__file__).parent

    # Auto-detect ONNX models and infer imgsz from input shape
    models = []
    for path in sorted(script_dir.glob("*.onnx")):
        session = ort.InferenceSession(str(path), providers=providers)
        input_shape = session.get_inputs()[0].shape
        imgsz = input_shape[2] if len(input_shape) == 4 else 1600
        print(f"Loading {path.name} (imgsz={imgsz})")
        models.append((session, imgsz, 1.0))

    if not models:
        raise FileNotFoundError("No model weights found")

    print(f"Ensemble: {len(models)} models loaded")

    image_files = sorted(
        f for f in Path(args.input).iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Processing {len(image_files)} images...")

    predictions = []
    for i, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        boxes, scores, labels = ensemble_predict(models, img, img_w, img_h)

        for box, score, label in zip(boxes, scores, labels):
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(float(x), 1) for x in box],
                "score": round(float(score), 4),
            })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(image_files)} done")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Done! {len(predictions)} predictions")


if __name__ == "__main__":
    main()
