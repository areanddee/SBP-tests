#!/bin/bash
# =============================================================================
# run_snapshot_comparison.sh — Run SBP vs Exact snapshot comparisons
# =============================================================================
# Usage:
#   ./run_snapshot_comparison.sh                    # face + vertex, N=48,96
#   ./run_snapshot_comparison.sh face               # face only
#   ./run_snapshot_comparison.sh vertex 48 96 192   # vertex at 3 resolutions
# =============================================================================

CENTER="${1:-both}"
shift
NS="${@:-48 96}"
if [ -z "$NS" ]; then NS="48 96"; fi

echo "=============================================="
echo "SBP vs Exact Snapshot Comparison"
echo "  Center: $CENTER"
echo "  N values: $NS"
echo "=============================================="

if [ "$CENTER" = "both" ] || [ "$CENTER" = "face" ]; then
    echo ""
    echo "=== FACE CENTER ==="
    python test_sbp_vs_exact_snapshots.py \
        --center face \
        --Ns $NS \
        --times 0.0 0.25 0.5 0.75 1.0
fi

if [ "$CENTER" = "both" ] || [ "$CENTER" = "vertex" ]; then
    echo ""
    echo "=== VERTEX CENTER ==="
    python test_sbp_vs_exact_snapshots.py \
        --center vertex \
        --Ns $NS \
        --times 0.0 0.25 0.5 0.75 1.0
fi

if [ "$CENTER" = "edge" ]; then
    echo ""
    echo "=== EDGE CENTER ==="
    python test_sbp_vs_exact_snapshots.py \
        --center edge \
        --Ns $NS \
        --times 0.0 0.25 0.5 0.75 1.0
fi

echo ""
echo "=============================================="
echo "Done. Check output directories:"
ls -d snapshots_* 2>/dev/null
echo "=============================================="
