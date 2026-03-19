#!/bin/bash
#SBATCH --job-name=package
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/package_%j.log

# Usage: sbatch package.slurm v22_full_x_1280
# Or:    sbatch package.slurm v22_full_x_1280 1600

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

RUN_NAME=${1:?Usage: sbatch package.slurm RUN_NAME [IMGSZ]}
IMGSZ_ARG=""
if [ -n "$2" ]; then
    IMGSZ_ARG="--imgsz $2"
fi

mkdir -p ${WORKDIR}/logs

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    ${CONTAINER} bash -c "
    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    cd ${WORKDIR}
    python3 package_model.py --run ${RUN_NAME} ${IMGSZ_ARG}
"
