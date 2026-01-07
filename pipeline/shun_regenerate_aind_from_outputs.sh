#!/bin/bash

# ============================================
# Regenerate AIND_* outputs from aind_output_scratch
# --------------------------------------------
# This script does NOT depend on aind_input/todo.
#
# Run on Kempner cluster with e.g.:
#   tmux new -s aind_regen
#   pipeline/shun_regenerate_aind_from_outputs.sh
#   # detach with:  Ctrl-b d
#   # reattach later: tmux attach -t aind_regen
#
# Optional usage:
#   pipeline/shun_regenerate_aind_from_outputs.sh pipeline/spike_sort.slrm
#   pipeline/shun_regenerate_aind_from_outputs.sh /path/to/aind_output_scratch
#
# Set FORCE=1 to delete/recreate existing AIND_* folders:
#   FORCE=1 pipeline/shun_regenerate_aind_from_outputs.sh
# ============================================

set -euo pipefail

ARG1="${1:-}"

# Default locations (override via slurm file or output dir argument)
DEFAULT_OUT_DIR="/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_scratch"
DOWNLOAD_BASE="/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_fordownload"

OUT_DIR="$DEFAULT_OUT_DIR"

if [ -n "$ARG1" ]; then
    if [ -f "$ARG1" ]; then
        # Assume it's a spike_sort.slrm; parse RESULTS_PATH
        SLURM_FILE_PATH=$(realpath "$ARG1")
        OUT_DIR=$(grep "^RESULTS_PATH" "$SLURM_FILE_PATH" | sed "s/RESULTS_PATH=//g" | tr -d '"')
        if [ -z "$OUT_DIR" ]; then
            echo "❌ Could not parse RESULTS_PATH from: $SLURM_FILE_PATH"
            exit 1
        fi
    else
        # Assume it's an output base dir
        OUT_DIR="$ARG1"
    fi
fi

OUT_DIR="${OUT_DIR%/}"

# Repo root (this script is in pipeline/)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT_DIR=$(dirname "$SCRIPT_DIR")
PY_SCRIPT="${REPO_ROOT_DIR%/}/postprocess/regenerate_aind_from_outputs.py"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ Python script not found: $PY_SCRIPT"
    exit 1
fi

echo "Regenerating AIND outputs from:"
echo "  OUT_DIR        : $OUT_DIR"
echo "  DOWNLOAD_BASE  : $DOWNLOAD_BASE"
echo "  PY_SCRIPT      : $PY_SCRIPT"
echo "  FORCE          : ${FORCE:-1}"

# Load cluster Python module and activate the mamba env "spikeinterface"
if command -v module >/dev/null 2>&1; then
    module load python || echo "⚠️ Failed to 'module load python'; please verify your module name."
fi

if command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook -s bash)" 2>/dev/null || true
    mamba activate spikeinterface || echo "⚠️ Failed to activate mamba env 'spikeinterface'; assuming environment is already correct."
else
    echo "⚠️ 'mamba' command not found; assuming correct Python environment already active."
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 not found in PATH; cannot run regeneration."
    exit 1
fi

echo ""
echo "Starting regeneration..."

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi

python3 "$PY_SCRIPT" \
    --output-base-dir "$OUT_DIR" \
    --download-base-dir "$DOWNLOAD_BASE" \
    $FORCE_FLAG

echo ""
echo "✅ Regeneration script finished."


