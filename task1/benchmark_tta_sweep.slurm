#!/bin/bash
#SBATCH --job-name=tta_sweep
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/tta_sweep_%j.log

# Sweep TTA + WBF parameters on the 3-model ensemble.
# Usage: sbatch benchmark_tta_sweep.slurm

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv
RUNS="v63_x_full v41_x_long200 v64_l_full"

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    --bind /cluster/projects/nn11127k:/cluster/projects/nn11127k \
    ${CONTAINER} bash -c "
    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    cd ${WORKDIR}

    echo '========================================='
    echo '  1. Baseline (no TTA, iou=0.6)'
    echo '========================================='
    python3 benchmark.py --runs ${RUNS} --ensemble --wbf-iou 0.6

    echo ''
    echo '========================================='
    echo '  2. TTA hflip (iou=0.6)'
    echo '========================================='
    python3 benchmark.py --runs ${RUNS} --ensemble --tta --wbf-iou 0.6

    echo ''
    echo '========================================='
    echo '  3. TTA hflip (iou=0.55)'
    echo '========================================='
    python3 benchmark.py --runs ${RUNS} --ensemble --tta --wbf-iou 0.55

    echo ''
    echo '========================================='
    echo '  4. TTA hflip (iou=0.5)'
    echo '========================================='
    python3 benchmark.py --runs ${RUNS} --ensemble --tta --wbf-iou 0.5

    echo ''
    echo '========================================='
    echo '  5. TTA hflip (iou=0.65)'
    echo '========================================='
    python3 benchmark.py --runs ${RUNS} --ensemble --tta --wbf-iou 0.65

    echo ''
    echo '========================================='
    echo '  6. TTA hflip + weighted models (iou=0.6)'
    echo '========================================='
    python3 benchmark.py --runs ${RUNS} --ensemble --tta --wbf-iou 0.6 --wbf-weights 1.2 1.0 0.8

    echo ''
    echo '========================================='
    echo '  SWEEP DONE'
    echo '========================================='
"
