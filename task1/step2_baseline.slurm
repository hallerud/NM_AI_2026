#!/bin/bash
#SBATCH --job-name=ngd_baseline
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/baseline_%j.log

# ============================================================
# STEP 2: CREATE BASELINE SUBMISSION
# Packages pretrained YOLOv8n + run.py into submission.zip
# ============================================================

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    ${CONTAINER} bash -c "
    export http_proxy=http://10.63.2.48:3128/
    export https_proxy=http://10.63.2.48:3128/

    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    cd ${WORKDIR}

    python3 << 'PYEOF'
import shutil
import zipfile
from pathlib import Path

WORKDIR = Path(\"${WORKDIR}\")
OUTPUT_DIR = WORKDIR / \"baseline_submission\"

print(\"=\" * 50)
print(\"  Creating Baseline Submission (YOLOv8n)\")
print(\"=\" * 50)

# Clean
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)
staging = OUTPUT_DIR / \"staging\"
staging.mkdir()

# ── 1. Create run.py ──
print(\"\\n1. Creating run.py...\")
run_code = '''import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--input\", required=True)
    parser.add_argument(\"--output\", required=True)
    args = parser.parse_args()

    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
    print(f\"Using device: {device}\")

    model = YOLO(\"yolov8n.pt\")
    predictions = []

    image_files = [f for f in sorted(Path(args.input).iterdir())
                   if f.suffix.lower() in (\".jpg\", \".jpeg\", \".png\")]
    print(f\"Processing {len(image_files)} images...\")

    for i, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split(\"_\")[-1])

        results = model(
            str(img_path),
            device=device,
            imgsz=1280,
            conf=0.15,
            iou=0.5,
            verbose=False,
        )

        for r in results:
            if r.boxes is None:
                continue
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                predictions.append({
                    \"image_id\": image_id,
                    \"category_id\": int(r.boxes.cls[j].item()),
                    \"bbox\": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    \"score\": round(float(r.boxes.conf[j].item()), 3),
                })

        if (i + 1) % 10 == 0:
            print(f\"  {i+1}/{len(image_files)} images done\")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, \"w\") as f:
        json.dump(predictions, f)
    print(f\"Done! {len(predictions)} predictions\")


if __name__ == \"__main__\":
    main()
'''

with open(staging / \"run.py\", \"w\") as f:
    f.write(run_code)
print(\"  Created run.py\")

# ── 2. Find and copy weights ──
print(\"\\n2. Finding yolov8n.pt...\")
import subprocess
result = subprocess.run([\"find\", str(Path.home()), \"-name\", \"yolov8n.pt\", \"-type\", \"f\"],
                       capture_output=True, text=True, timeout=10)
candidates = [l.strip() for l in result.stdout.strip().split(\"\\n\") if l.strip()]

for extra in [\"yolov8n.pt\", str(WORKDIR / \"yolov8n.pt\")]:
    if Path(extra).exists() and extra not in candidates:
        candidates.insert(0, extra)

if not candidates:
    print(\"  Downloading yolov8n.pt...\")
    from ultralytics import YOLO
    YOLO(\"yolov8n.pt\")
    result = subprocess.run([\"find\", str(Path.home()), \"-name\", \"yolov8n.pt\", \"-type\", \"f\"],
                           capture_output=True, text=True, timeout=10)
    candidates = [l.strip() for l in result.stdout.strip().split(\"\\n\") if l.strip()]

weights = Path(candidates[0])
shutil.copy2(weights, staging / \"yolov8n.pt\")
size_mb = (staging / \"yolov8n.pt\").stat().st_size / 1024 / 1024
print(f\"  Copied: {weights} ({size_mb:.1f} MB)\")

# ── 3. Create zip ──
print(\"\\n3. Creating submission.zip...\")
zip_path = OUTPUT_DIR / \"submission.zip\"
with zipfile.ZipFile(zip_path, \"w\", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(staging.iterdir()):
        zf.write(f, f.name)
        print(f\"  Added: {f.name}\")

# ── 4. Verify ──
print(\"\\n4. Verify:\")
with zipfile.ZipFile(zip_path, \"r\") as zf:
    for info in zf.infolist():
        print(f\"  {info.filename:30s} {info.file_size / 1024 / 1024:8.1f} MB\")
    assert \"run.py\" in zf.namelist(), \"ERROR: run.py not at zip root!\"
    print(\"  run.py at zip root\")

total = zip_path.stat().st_size / 1024 / 1024
print(f\"  Total: {total:.1f} MB {'OK' if total <= 420 else 'OVER LIMIT'}\")

print(f\"\\n{'=' * 50}\")
print(f\"  DONE! File ready at:\")
print(f\"  {zip_path}\")
print(f\"{'=' * 50}\")
PYEOF
"
