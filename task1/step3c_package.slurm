#!/bin/bash
#SBATCH --job-name=ngd_package
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/package_%j.log

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
    python3 ${WORKDIR}/package_submission.py
"
