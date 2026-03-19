#!/bin/bash
#SBATCH --job-name=ngd_convert
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/convert_%j.log

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    ${CONTAINER} bash -c "
    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    cd ${WORKDIR}
    python3 ${WORKDIR}/convert_dataset.py
"
