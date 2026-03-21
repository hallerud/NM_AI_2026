#!/bin/bash
# Phase 4: Proper val split + product images
# All runs use yolo_split for honest eval. Best config retrained on yolo_full for submission.
# Run: bash launch_sweep4.sh

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

# v50-v52: eval runs on yolo_split (223 train + 1577 product / 25 val)
# v53-v55: eval runs on yolo_split WITHOUT product images for comparison
# FORMAT: NAME MODEL IMGSZ EPOCHS BOX CLS CP CM MOSAIC MIXUP ERASING DEG SCALE LR0 LRF WARMUP MS DATASET
EXPERIMENTS=(
    "v50_x_prod1600       yolov8x.pt  1600  150  7.5  0.5  0.4  10  1.0  0.1  0.4  5.0   0.5   0.001   0.01   5  0.0  yolo_split"
    "v51_x_prod_ms        yolov8x.pt  1600  150  7.5  0.5  0.4  10  1.0  0.1  0.4  5.0   0.5   0.001   0.01   5  0.5  yolo_split"
    "v52_x_prod_heavy     yolov8x.pt  1600  150  7.5  0.5  0.6  15  1.0  0.2  0.5  10.0  0.6   0.001   0.01   5  0.0  yolo_split"
    "v53_x_noprod1600     yolov8x.pt  1600  150  7.5  0.5  0.4  10  1.0  0.1  0.4  5.0   0.5   0.001   0.01   5  0.0  yolo_split_noprod"
    "v54_l_prod1600       yolov8l.pt  1600  150  7.5  0.5  0.4  10  1.0  0.1  0.4  5.0   0.5   0.001   0.01   5  0.0  yolo_split"
    "v55_x_prod_long      yolov8x.pt  1600  200  7.5  0.5  0.4  15  1.0  0.1  0.4  5.0   0.5   0.001   0.005  5  0.0  yolo_split"
)

# Create the no-product dataset for comparison
echo "Creating no-product split dataset for comparison..."
cd ${WORKDIR}
python3 make_dataset.py  # without --product_images, writes to yolo_split
# Rename it
if [ -d yolo_split_noprod ]; then rm -rf yolo_split_noprod; fi
mv yolo_split yolo_split_noprod
# Recreate with product images
python3 make_dataset.py --product_images
echo ""

echo "Submitting ${#EXPERIMENTS[@]} Phase 4 experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    read -r NAME MODEL IMGSZ EPOCHS BOX CLS CP CM MOSAIC MIXUP ERASING DEG SCALE LR0 LRF WARMUP MS DATASET <<< "$exp"

    if [ "$EPOCHS" -gt 150 ]; then
        WALLTIME="10:00:00"
    else
        WALLTIME="08:00:00"
    fi

    JOB_SCRIPT=$(mktemp /tmp/sweep4_XXXXXX.slurm)
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
        --degrees ${DEG} \\
        --scale ${SCALE} \\
        --lr0 ${LR0} \\
        --lrf ${LRF} \\
        --warmup_epochs ${WARMUP} \\
        --multi_scale ${MS} \\
        --dataset ${DATASET}
"
SLURM

    JOBID=$(sbatch ${JOB_SCRIPT} 2>&1)
    echo "  ${NAME} (${DATASET}): ${JOBID}"
    rm ${JOB_SCRIPT}
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Compare: python3 compare_runs.py"
echo ""
echo "Key comparisons:"
echo "  v50 vs v53  → does product images help?"
echo "  v50 vs v51  → does multi-scale help?"
echo "  v50 vs v52  → does heavy aug help?"
echo "  v50 vs v55  → does longer training help?"
