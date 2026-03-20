#!/bin/bash
#SBATCH --job-name=pkg_ensemble
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/package_ensemble_%j.log

# Usage: sbatch package_ensemble.slurm "v63_x_full v62_rtdetr_full"

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

ARGS=${1:?Usage: sbatch package_ensemble.slurm "--runs RUN1 RUN2 [--half 1 1] [--out DIR]"}

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    --bind /cluster/projects/nn11127k:/cluster/projects/nn11127k \
    ${CONTAINER} bash -c "
    source ${VENV}/bin/activate
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    cd ${WORKDIR}
    python3 package_ensemble.py ${ARGS}
"
