#!/bin/bash
# Launch RT-DETR eval + full runs with correct defaults.
# Run: bash launch_rtdetr.sh

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

EXPERIMENTS=(
    "v60_rtdetr_eval    rtdetr-l.pt  1600  150  30  yolo_split  06:00:00"
    "v62_rtdetr_full    rtdetr-l.pt  1600  150  30  yolo_full   06:00:00"
)

echo "Relaunching RT-DETR with fixed config..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    read -r NAME MODEL IMGSZ EPOCHS PATIENCE DATASET WALLTIME <<< "$exp"

    JOB_SCRIPT=$(mktemp /tmp/rtdetr_XXXXXX.slurm)
    cat > ${JOB_SCRIPT} << SLURM
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=${WALLTIME}
#SBATCH --output=${WORKDIR}/logs/${NAME}_%j.log

apptainer exec --nv \\
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \\
    --bind /cluster/projects/nn11127k:/cluster/projects/nn11127k \\
    ${CONTAINER} bash -c "
    export http_proxy=http://10.63.2.48:3128/
    export https_proxy=http://10.63.2.48:3128/
    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\\\${PYTHONPATH}
    cd ${WORKDIR}
    python3 sweep_train.py \\
        --name ${NAME} \\
        --model ${MODEL} \\
        --imgsz ${IMGSZ} \\
        --epochs ${EPOCHS} \\
        --patience ${PATIENCE} \\
        --lr0 0.0001 \\
        --lrf 0.01 \\
        --warmup_epochs 3 \\
        --dataset ${DATASET}
"
SLURM

    JOBID=$(sbatch ${JOB_SCRIPT} 2>&1)
    echo "  ${NAME}: ${JOBID}"
    rm ${JOB_SCRIPT}
done

echo ""
echo "Monitor: squeue -u \$USER"
