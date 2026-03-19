#!/bin/bash
# Launch all sweep experiments as separate SLURM jobs.
# Each gets 1 GPU. Run: bash launch_sweep.sh

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

mkdir -p ${WORKDIR}/logs

# Step 1: Submit dataset conversion as a quick GPU job
CONVERT_SCRIPT=$(mktemp /tmp/convert_XXXXXX.slurm)
cat > ${CONVERT_SCRIPT} << 'CONVERTEOF'
#!/bin/bash
#SBATCH --job-name=convert_full
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/home/ksv023/NM_AI_2026/task1/logs/convert_full_%j.log

WORKDIR=/cluster/home/ksv023/NM_AI_2026/task1
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
VENV=${WORKDIR}/venv

apptainer exec --nv \
    --bind /cluster/home/ksv023:/cluster/home/ksv023 \
    ${CONTAINER} bash -c "
    source ${VENV}/bin/activate
    export PYTHONPATH=${WORKDIR}/pylibs:\${PYTHONPATH}
    python3 ${WORKDIR}/convert_fulldata.py
"
CONVERTEOF

CONVERT_JOB=$(sbatch --parsable ${CONVERT_SCRIPT})
echo "  Convert job: ${CONVERT_JOB}"
rm ${CONVERT_SCRIPT}

# Define experiments: NAME MODEL IMGSZ EPOCHS BOX CLS COPY_PASTE
# Each line = one SLURM job (all depend on convert job)
EXPERIMENTS=(
    # Exp 1: baseline but full data + more epochs (control)
    "v20_full_l_1280    yolov8l.pt  1280  150  7.5  0.5  0.2"
    # Exp 2: higher resolution
    "v21_full_l_1600    yolov8l.pt  1600  150  7.5  0.5  0.2"
    # Exp 3: bigger model
    "v22_full_x_1280    yolov8x.pt  1280  150  7.5  0.5  0.2"
    # Exp 4: bigger model + higher res
    "v23_full_x_1600    yolov8x.pt  1600  150  7.5  0.5  0.2"
    # Exp 5: emphasize detection loss (70% of competition score)
    "v24_full_l_highbox  yolov8l.pt  1280  150  12.0  0.3  0.2"
    # Exp 6: more augmentation
    "v25_full_l_moreaug  yolov8l.pt  1280  150  7.5  0.5  0.4"
)

echo ""
echo "Submitting ${#EXPERIMENTS[@]} experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    read -r NAME MODEL IMGSZ EPOCHS BOX CLS CP <<< "$exp"

    JOB_SCRIPT=$(mktemp /tmp/sweep_XXXXXX.slurm)
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
        --dataset yolo_dataset_full
"
SLURM

    JOBID=$(sbatch --dependency=afterok:${CONVERT_JOB} ${JOB_SCRIPT} 2>&1)
    echo "  ${NAME}: ${JOBID}"
    rm ${JOB_SCRIPT}
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${WORKDIR}/logs/v2*"
