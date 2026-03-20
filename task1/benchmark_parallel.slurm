#!/bin/bash
#SBATCH --job-name=bm_tta
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/bm_tta_%j.log

# Generic single-config benchmark job.
# Usage: sbatch benchmark_parallel.slurm "--runs v63_x_full v41_x_long200 v64_l_full --ensemble --tta --wbf-iou 0.55"

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

ARGS=${1:?Usage: sbatch benchmark_parallel.slurm "BENCHMARK_ARGS"}

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    --bind /cluster/projects/nn11127k:/cluster/projects/nn11127k \
    ${CONTAINER} bash -c "
    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    cd ${WORKDIR}
    python3 benchmark.py ${ARGS}
"
