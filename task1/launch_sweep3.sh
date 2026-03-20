#!/bin/bash
# Phase 3: Focus on generalization (closing the val→test gap)
# Key changes: multi-scale training, heavier aug, longer close_mosaic, varied architectures
# Run: bash launch_sweep3.sh

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

# Strategy overview:
# v40: Best config (v32) + multi-scale training for robustness
# v41: Best config + longer epochs (200) + cosine restart via lower lrf
# v42: YOLOv8x 1600 + heavy augmentation (more copy_paste, mixup, erasing)
# v43: YOLOv8x 1280 + multi-scale (cheaper, faster convergence)
# v44: YOLOv8l 1600 + heavy aug + multi-scale (diverse architecture for ensemble)
# v45: YOLOv8x 1600 + conservative aug (less aug, more fitting to data)

# FORMAT: NAME MODEL IMGSZ EPOCHS BOX CLS CP CM MOSAIC MIXUP ERASING DEGREES SCALE LR0 LRF WARMUP MULTI_SCALE
EXPERIMENTS=(
    "v40_x_multiscale     yolov8x.pt  1600  150  7.5  0.5  0.4  10  1.0  0.1  0.4  5.0  0.5  0.001  0.01  5  0.5"
    "v41_x_long200        yolov8x.pt  1600  200  7.5  0.5  0.4  15  1.0  0.1  0.4  5.0  0.5  0.001  0.005  5  0.0"
    "v42_x_heavyaug       yolov8x.pt  1600  150  7.5  0.5  0.6  15  1.0  0.2  0.5  10.0  0.6  0.001  0.01  5  0.0"
    "v43_x_ms1280         yolov8x.pt  1280  150  7.5  0.5  0.4  10  1.0  0.1  0.4  5.0  0.5  0.001  0.01  5  0.5"
    "v44_l_heavyaug_ms    yolov8l.pt  1600  150  7.5  0.5  0.6  15  1.0  0.2  0.5  10.0  0.6  0.001  0.01  5  0.5"
    "v45_x_conserv        yolov8x.pt  1600  200  7.5  0.5  0.2  20  0.8  0.0  0.2  3.0  0.4  0.0005  0.01  3  0.0"
)

echo "Submitting ${#EXPERIMENTS[@]} Phase 3 experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    read -r NAME MODEL IMGSZ EPOCHS BOX CLS CP CM MOSAIC MIXUP ERASING DEGREES SCALE LR0 LRF WARMUP MS <<< "$exp"

    # 8h for 200-epoch runs, 6h for 150-epoch
    if [ "$EPOCHS" -gt 150 ]; then
        WALLTIME="10:00:00"
    else
        WALLTIME="08:00:00"
    fi

    JOB_SCRIPT=$(mktemp /tmp/sweep3_XXXXXX.slurm)
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
        --box ${BOX} \\
        --cls ${CLS} \\
        --copy_paste ${CP} \\
        --close_mosaic ${CM} \\
        --mosaic ${MOSAIC} \\
        --mixup ${MIXUP} \\
        --erasing ${ERASING} \\
        --degrees ${DEGREES} \\
        --scale ${SCALE} \\
        --lr0 ${LR0} \\
        --lrf ${LRF} \\
        --warmup_epochs ${WARMUP} \\
        --multi_scale ${MS} \\
        --dataset yolo_dataset_full
"
SLURM

    JOBID=$(sbatch ${JOB_SCRIPT} 2>&1)
    echo "  ${NAME}: ${JOBID}"
    rm ${JOB_SCRIPT}
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${WORKDIR}/logs/v4*"
