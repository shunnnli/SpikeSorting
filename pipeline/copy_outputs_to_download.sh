#!/bin/bash

# ============================================
# Copy all session output folders from aind_output_scratch to aind_output_fordownload
# --------------------------------------------
# Run on Kempner cluster with:
#   bash pipeline/copy_outputs_to_download.sh
#
# Or specify custom paths:
#   bash pipeline/copy_outputs_to_download.sh /path/to/scratch /path/to/download
# ============================================

# Default paths
OUTPUT_SCRATCH="${1:-/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_scratch}"
DOWNLOAD_BASE="${2:-/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_fordownload}"

OUTPUT_SCRATCH="${OUTPUT_SCRATCH%/}"
DOWNLOAD_BASE="${DOWNLOAD_BASE%/}"

echo "Copying session output folders from:"
echo "  Source: $OUTPUT_SCRATCH"
echo "  Destination: $DOWNLOAD_BASE"
echo ""

if [ ! -d "$OUTPUT_SCRATCH" ]; then
    echo "❌ Source directory not found: $OUTPUT_SCRATCH"
    exit 1
fi

mkdir -p "$DOWNLOAD_BASE"

# Find all *_output folders in the scratch directory
session_folders=("$OUTPUT_SCRATCH"/*_output)
if [ ! -e "${session_folders[0]}" ]; then
    echo "⚠️  No *_output folders found in: $OUTPUT_SCRATCH"
    exit 0
fi

copied=0
skipped=0

for src_folder in "${session_folders[@]}"; do
    # Skip if not a directory (glob didn't match)
    [ -d "$src_folder" ] || continue
    
    folder_name=$(basename "$src_folder")
    dst_folder="${DOWNLOAD_BASE}/${folder_name}"
    
    if [ -d "$src_folder" ]; then
        echo "  Copying $folder_name -> $DOWNLOAD_BASE/ (following symlinks)"
        # Use -rL to follow symlinks so that the *contents* of linked folders
        # are copied, not the symlinks themselves. This makes the download
        # directory self-contained on your local machine.
        cp -rL "$src_folder" "$DOWNLOAD_BASE/"
        if [ $? -eq 0 ]; then
            copied=$((copied + 1))
        else
            echo "    ❌ Failed to copy $folder_name"
            skipped=$((skipped + 1))
        fi
    else
        echo "  Skipping $folder_name: not a directory"
        skipped=$((skipped + 1))
    fi
done

echo ""
echo "✅ Finished copying session output folders."
echo "  Copied: $copied"
echo "  Skipped/Failed: $skipped"

