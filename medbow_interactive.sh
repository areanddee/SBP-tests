#!/bin/bash
# =============================================================================
# medicinebow_interactive.sh
# Get an interactive GPU session on Wyoming ARCC Medicinebow cluster
# =============================================================================
#
# Usage:
#   ./medicinebow_interactive.sh [gpu_type] [hours]
#
# Arguments:
#   gpu_type: h100 (default), l40s, a30, a6000
#   hours:    session duration (default: 2)
#
# Examples:
#   ./medicinebow_interactive.sh           # 2-hour H100 session
#   ./medicinebow_interactive.sh l40s      # 2-hour L40S session (recommended)
#   ./medicinebow_interactive.sh h100 4    # 4-hour H100 session
#   ./medicinebow_interactive.sh a6000     # 2-hour A6000 session
#
# Note: L40S (Lovelace) is recommended due to cuDNN issues with H100 + bidirectional LSTMs
# =============================================================================
GPU_TYPE="${1:-h100}"
HOURS="${2:-2}"
ACCOUNT="atm-jax"

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
echo "Medicinebow Interactive GPU Session"
echo "=============================================="
echo "GPU type:  $GPU_TYPE"
echo "Duration:  $HOURS hours"
echo "Account:   $ACCOUNT"
echo "Partition: $PARTITION"
echo "=============================================="
# Request interactive session
echo "Requesting $GPU_TYPE GPU..."
echo ""
srun --partition=$PARTITION \
     --account=$ACCOUNT \
     --gres=gpu:$GPU_TYPE:3 \
     --ntasks=1 \
     --mem-per-gpu=128G \
     --time=${HOURS}:00:00 \
     --pty bash
