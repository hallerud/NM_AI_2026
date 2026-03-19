#!/bin/bash
# Phase 2b sweep: build on best performers (v25, v22, v24).
# Dataset already exists — no convert step needed.
# Run: bash launch_sweep2.sh

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

# NAME MODEL IMGSZ EPOCHS BOX CLS CP CLOSE_MOSAIC
EXPERIMENTS=(
    # Combine best model (x) with best aug (copy_paste=0.4)
    "v30_x_moreaug       yolov8x.pt  1280  150  7.5   0.5  0.4  10"
    # Push augmentation further
    "v31_l_heavyaug      yolov8l.pt  1280  150  7.5   0.5  0.6  10"
    # Best combo at higher resolution
    "v32_x_moreaug_1600  yolov8x.pt  1600  150  7.5   0.5  0.4  10"
    # Combine v25 aug + v24 loss weighting
    "v33_l_combo         yolov8l.pt  1280  150  12.0  0.3  0.4  10"
    # Keep mosaic on longer before disabling
    "v34_l_longmosaic    yolov8l.pt  1280  150  7.5   0.5  0.4  20"
    # Full combo: big model + best aug + detection focus
    "v35_x_combo         yolov8x.pt  1280  150  12.0  0.3  0.4  10"
)

echo "Submitting ${#EXPERIMENTS[@]} experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    read -r NAME MODEL IMGSZ EPOCHS BOX CLS CP CM <<< "$exp"

    JOB_SCRIPT=$(mktemp /tmp/sweep2_XXXXXX.slurm)
    cat > ${JOB_SCRIPT} << SLURM
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=${WORKDIR}/logs/${NAME}_%j.log

apptainer exec --nv \\
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \\
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
        --box ${BOX} \\
        --cls ${CLS} \\
        --copy_paste ${CP} \\
        --close_mosaic ${CM} \\
        --dataset yolo_dataset_full
"
SLURM

    JOBID=$(sbatch ${JOB_SCRIPT} 2>&1)
    echo "  ${NAME}: ${JOBID}"
    rm ${JOB_SCRIPT}
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${WORKDIR}/logs/v3*"
