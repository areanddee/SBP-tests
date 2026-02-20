#!/bin/bash
# =============================================================================
# medbow_batch.sh
# Submit a batch GPU job on Wyoming ARCC Medicinebow cluster
# =============================================================================
#
# Usage:
#   ./medbow_batch.sh [gpu_type] [hours] [script] [args...]
#
# Arguments:
#   gpu_type: h100 (default), l40s, a30, a6000
#   hours:    session duration (default: 2)
#   script:   python script to run (default: test_stag_step5_Nconv.py)
#   args:     additional arguments passed to the script
#
# Examples:
#   ./medbow_batch.sh h100 2 test_stag_step5_Nconv.py --gauss 1
#   ./medbow_batch.sh a30 1 test_stag_step5_Nconv.py --gauss 2
#   ./medbow_batch.sh h100 3 test_stag_step5_Nconv.py
#
# Output: slurm-<jobid>.out in current directory
# =============================================================================

GPU_TYPE="${1:-h100}"
HOURS="${2:-2}"
SCRIPT="${3:-test_stag_step5_Nconv.py}"
shift 3 2>/dev/null
SCRIPT_ARGS="$@"

ACCOUNT="atm-jax"
JOB_NAME="sbp-${GPU_TYPE}-$(basename ${SCRIPT} .py)"

# Map GPU type to correct partition
case "$GPU_TYPE" in
    l40s)
        PARTITION="mb-l40s"
        ;;
    h100)
        PARTITION="mb-h100"
        ;;
    a30)
        PARTITION="mb-a30"
        ;;
    a6000)
        PARTITION="mb-a6000"
        ;;
    *)
        echo "Error: Unknown GPU type '$GPU_TYPE'"
        echo "Valid options: h100, l40s, a30, a6000"
        exit 1
        ;;
esac

echo "=============================================="
echo "Medicinebow Batch GPU Submission"
echo "=============================================="
echo "GPU type:  $GPU_TYPE"
echo "Duration:  $HOURS hours"
echo "Account:   $ACCOUNT"
echo "Partition: $PARTITION"
echo "Script:    $SCRIPT $SCRIPT_ARGS"
echo "Job name:  $JOB_NAME"
echo "=============================================="

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=12
#SBATCH --mem-per-gpu=128G
#SBATCH --time=${HOURS}:00:00
#SBATCH --output=%x-%j.out

echo "=============================================="
echo "Job: \${SLURM_JOB_NAME} (ID: \${SLURM_JOB_ID})"
echo "Node: \$(hostname)"
echo "GPU:  \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: \$(date)"
echo "Dir:  \$(pwd)"
echo "=============================================="

# Activate conda environment
source activate jax-sbp 2>/dev/null || conda activate jax-sbp 2>/dev/null

cd ${PWD}
python ${SCRIPT} ${SCRIPT_ARGS}

echo ""
echo "=============================================="
echo "Finished: \$(date)"
echo "=============================================="
EOF
