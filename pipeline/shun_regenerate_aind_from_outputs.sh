#!/bin/bash

# ============================================
# Regenerate AIND_* outputs from aind_output_scratch
# --------------------------------------------
# This script does NOT depend on aind_input/todo.
# Uses extract_aind_output.py with --session for each session folder.
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
# FORCE defaults to 1 (delete/recreate existing AIND_* folders).
# Set FORCE=0 to skip existing folders:
#   FORCE=0 pipeline/shun_regenerate_aind_from_outputs.sh
#
# NO_COPY=1 to skip copying to download directory:
#   NO_COPY=1 pipeline/shun_regenerate_aind_from_outputs.sh
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
PY_SCRIPT="${REPO_ROOT_DIR%/}/postprocess/extract_aind_output.py"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ Python script not found: $PY_SCRIPT"
    exit 1
fi

echo "Regenerating AIND outputs from:"
echo "  OUT_DIR        : $OUT_DIR"
echo "  DOWNLOAD_BASE  : $DOWNLOAD_BASE"
echo "  PY_SCRIPT      : $PY_SCRIPT"
echo "  FORCE          : ${FORCE:-1}"
echo "  NO_COPY        : ${NO_COPY:-0}"

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

# Create download directory if needed
mkdir -p "$DOWNLOAD_BASE"

# Find all *_output session folders
SESSION_DIRS=$(find "$OUT_DIR" -maxdepth 1 -type d -name "*_output" | sort)

if [ -z "$SESSION_DIRS" ]; then
    echo "❌ No *_output folders found under: $OUT_DIR"
    exit 1
fi

OK=0
SKIPPED=0
FAILED=0

for SESSION_DIR in $SESSION_DIRS; do
    SESSION_DIR_NAME=$(basename "$SESSION_DIR")
    # Strip _output suffix to get session name
    SESSION_NAME="${SESSION_DIR_NAME%_output}"
    AIND_FOLDER="$SESSION_DIR/AIND_$SESSION_NAME"
    
    echo ""
    echo "=== Session folder: $SESSION_DIR_NAME  (session_name=$SESSION_NAME) ==="
    
    # If FORCE=1, delete existing AIND folder
    if [ "${FORCE:-1}" = "1" ] && [ -d "$AIND_FOLDER" ]; then
        echo "  Removing existing AIND folder: $AIND_FOLDER"
        rm -rf "$AIND_FOLDER"
    fi
    
    # Run extract_aind_output.py with --session
    if python3 "$PY_SCRIPT" --session "$SESSION_DIR"; then
        OK=$((OK + 1))
        
        # Copy to download directory if NO_COPY is not set
        if [ "${NO_COPY:-0}" != "1" ] && [ -d "$AIND_FOLDER" ]; then
            SESSION_DOWNLOAD_DIR="$DOWNLOAD_BASE/${SESSION_NAME}_output"
            DST="$SESSION_DOWNLOAD_DIR/AIND_$SESSION_NAME"
            echo "  Copying AIND folder -> $DST"
            mkdir -p "$SESSION_DOWNLOAD_DIR"
            # Remove existing and copy fresh
            rm -rf "$DST"
            cp -rL "$AIND_FOLDER" "$DST"  # -L follows symlinks
        fi
    else
        echo "  [error] Failed to process session: $SESSION_DIR_NAME"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Done."
echo "  regenerated_ok=$OK, skipped=$SKIPPED, failed=$FAILED"

if [ "$FAILED" -gt 0 ]; then
    exit 2
fi

echo ""
echo "✅ Regeneration script finished."


