#!/bin/bash
#SBATCH --job-name=ngd_setup
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_%j.log

# ============================================================
# STEP 1: SETUP
# Creates venv, installs ultralytics, downloads model weights.
# ============================================================

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    --bind /cluster/projects/nn11127k:/cluster/projects/nn11127k \
    ${CONTAINER} bash -c "
    export http_proxy=http://10.63.2.48:3128/
    export https_proxy=http://10.63.2.48:3128/

    cd ${WORKDIR}

    echo '============================================'
    echo '  NorgesGruppen - Step 1: Setup'
    echo '============================================'
    echo ''

    # ── Create venv with system site-packages (to get torch) ──
    if [ ! -d ${VENV} ]; then
        echo 'Creating virtual environment...'
        python3 -m venv --system-site-packages ${VENV}
    fi
    source ${VENV}/bin/activate

    # ── Install compatible packages ──
    PYLIBS=${WORKDIR}/pylibs
    mkdir -p \${PYLIBS}

    echo 'Installing numpy<2 to custom path (override container numpy 2.4.3)...'
    pip install --target=\${PYLIBS} --quiet 'numpy==1.26.4'
    export PYTHONPATH=\${PYLIBS}:\${PYTHONPATH}

    echo 'Installing ultralytics...'
    pip install --quiet 'ultralytics>=8.3.0'
    echo 'Fixing opencv (headless)...'
    pip uninstall -y opencv-python 2>/dev/null
    pip install --quiet --force-reinstall opencv-python-headless
    echo ''

    # ── Verify imports ──
    echo 'Verifying imports...'
    python3 -c '
import torch
print(f\"  PyTorch: {torch.__version__}\")
print(f\"  CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"  GPU: {torch.cuda.get_device_name(0)}\")
    print(f\"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")
import numpy as np
print(f\"  NumPy: {np.__version__}\")
from ultralytics import YOLO
print(f\"  Ultralytics imported OK\")
'
    echo ''

    # ── Check data ──
    echo 'Looking for annotations.json...'
    find ${WORKDIR} -name 'annotations.json' -type f 2>/dev/null
    echo ''

    echo 'Looking for training images...'
    IMG_COUNT=\$(find ${WORKDIR} -name 'img_*.jpg' -o -name 'img_*.jpeg' 2>/dev/null | wc -l)
    echo \"Found \${IMG_COUNT} training images\"
    echo ''

    # ── Download YOLOv8 pretrained weights ──
    echo 'Downloading YOLOv8n weights (for baseline)...'
    python3 -c '
from ultralytics import YOLO
m = YOLO(\"yolov8n.pt\")
print(\"  yolov8n.pt ready\")
'

    echo ''
    echo 'Downloading YOLOv8l weights (for training)...'
    python3 -c '
from ultralytics import YOLO
m = YOLO(\"yolov8l.pt\")
print(\"  yolov8l.pt ready\")
'

    echo ''
    echo '============================================'
    echo '  Setup complete!'
    echo ''
    echo '  Data: ${WORKDIR}/data/train/annotations.json'
    echo '  Images: ${WORKDIR}/data/train/images/'
    echo '  Venv: ${VENV}'
    echo ''
    echo '  Next: sbatch step2_baseline.slurm'
    echo '============================================'
"
