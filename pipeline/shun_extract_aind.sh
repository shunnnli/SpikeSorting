#!/bin/bash

# ============================================
# Batch AIND export for existing spike-sorted data
# --------------------------------------------
# Run on Kempner cluster with e.g.:
#   tmux new -s aind_export
#   pipeline/shun_extract_aind.sh pipeline/spike_sort.slrm
#   # detach with:  Ctrl-b d
#   # reattach later: tmux attach -t aind_export
# ============================================

# First and only argument: path to the spike-sorting Slurm file
SLURM_FILE="$1"

if [ -z "$SLURM_FILE" ]; then
    echo "Usage: $0 <path/to/spike_sort.slrm>"
    exit 1
fi

# Resolve the Slurm file to an absolute path
SLURM_FILE_PATH=$(realpath "$SLURM_FILE")

if [ ! -f "$SLURM_FILE_PATH" ]; then
    echo "❌ Slurm file not found: $SLURM_FILE_PATH"
    exit 1
fi

echo "Using Slurm configuration from: $SLURM_FILE_PATH"

# Extract relevant paths from the Slurm file
TOP_DIR=$(grep "^DATA_PATH" "$SLURM_FILE_PATH" | sed "s/DATA_PATH=//g" | tr -d '"')
OUT_DIR=$(grep "^RESULTS_PATH" "$SLURM_FILE_PATH" | sed "s/RESULTS_PATH=//g" | tr -d '"')
PIPELINE_PATH=$(grep "^PIPELINE_PATH" "$SLURM_FILE_PATH" | sed "s/PIPELINE_PATH=//g" | tr -d '"')

if [ -z "$TOP_DIR" ] || [ -z "$OUT_DIR" ] || [ -z "$PIPELINE_PATH" ]; then
    echo "❌ Failed to parse DATA_PATH / RESULTS_PATH / PIPELINE_PATH from $SLURM_FILE_PATH"
    exit 1
fi

PIPELINE_CODE_DIR=$(realpath "$PIPELINE_PATH")
REPO_ROOT_DIR=$(dirname "$PIPELINE_CODE_DIR")
POSTPROCESS_SCRIPT_PATH="${REPO_ROOT_DIR%/}/postprocess/extract_aind_output.py"

echo "Input (raw) base directory   : $TOP_DIR"
echo "Spike-sorted output base dir : $OUT_DIR"
echo "Pipeline code directory      : $PIPELINE_CODE_DIR"
echo "Postprocess script           : $POSTPROCESS_SCRIPT_PATH"

if [ ! -f "$POSTPROCESS_SCRIPT_PATH" ]; then
    echo "❌ Post-processing script not found at: $POSTPROCESS_SCRIPT_PATH"
    echo "Please verify that postprocess/extract_aind_output.py exists."
    exit 1
fi

# Configure environment variables for extract_aind_output.py
export AIND_INPUT_BASE_DIR="$TOP_DIR"
export AIND_OUTPUT_BASE_DIR="$OUT_DIR"

echo ""
echo "Environment for extract_aind_output.py:"
echo "  AIND_INPUT_BASE_DIR = $AIND_INPUT_BASE_DIR"
echo "  AIND_OUTPUT_BASE_DIR = $AIND_OUTPUT_BASE_DIR"

# Load cluster Python module and activate the mamba env "spikeinterface"
if command -v module >/dev/null 2>&1; then
    module load python || echo "⚠️ Failed to 'module load python'; please verify your module name."
fi

if command -v mamba >/dev/null 2>&1; then
    # Initialize mamba in this non-interactive shell and activate env
    eval "$(mamba shell hook -s bash)" 2>/dev/null || true
    mamba activate spikeinterface || echo "⚠️ Failed to activate mamba env 'spikeinterface'; assuming environment is already correct."
else
    echo "⚠️ 'mamba' command not found; assuming correct Python environment already active."
fi

# Use python3 explicitly for post-processing
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 not found in PATH; cannot run post-processing."
    exit 1
fi

echo ""
echo "Starting AIND export for all sessions found under:"
echo "  raw input   : $AIND_INPUT_BASE_DIR"
echo "  spike output: $AIND_OUTPUT_BASE_DIR"

python3 "$POSTPROCESS_SCRIPT_PATH"
POST_STATUS=$?

if [ $POST_STATUS -ne 0 ]; then
    echo "❌ extract_aind_output.py exited with status $POST_STATUS"
    exit $POST_STATUS
fi

echo "✅ AIND export completed successfully for all matching sessions."

# ============================================
# Copy AIND_* folders to download directory
# ============================================
echo ""
DOWNLOAD_BASE="/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_fordownload"
echo "Copying AIND_* folders to: $DOWNLOAD_BASE"
mkdir -p "$DOWNLOAD_BASE"

for session_dir in "${OUT_DIR%/}"/*_output; do
    # Skip if glob didn't match anything
    [ -d "$session_dir" ] || continue

    session_name=$(basename "$session_dir")
    echo "  Checking session folder: $session_name"

    for aind_dir in "$session_dir"/AIND_*; do
        [ -d "$aind_dir" ] || continue
        echo "    Copying $(basename "$aind_dir") -> $DOWNLOAD_BASE/ (following symlinks)"
        # -rL: recursive, follow symlinks so the download directory is self-contained
        cp -rL "$aind_dir" "$DOWNLOAD_BASE/"
    done
done

echo "✅ Finished copying all available AIND_* folders to $DOWNLOAD_BASE"

