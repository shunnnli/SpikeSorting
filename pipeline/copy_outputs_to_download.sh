#!/bin/bash

############################################################
# Copy all session output folders from aind_output_scratch
# to aind_output_fordownload
# ----------------------------------------------------------
# Run on Kempner cluster with:
#   bash pipeline/copy_outputs_to_download.sh
#
# Or specify custom paths:
#   bash pipeline/copy_outputs_to_download.sh /path/to/scratch /path/to/download
#
# Or specify a user profile (uses pipeline/user_profiles.conf):
#   bash pipeline/copy_outputs_to_download.sh \"\" \"\" <user_profile>
#
# If <user_profile> is provided, OUTPUT_SCRATCH and DOWNLOAD_BASE
# are derived from that profile:
#   base       = value from user_profiles.conf OR
#                /n/netscratch/bsabatini_lab/Lab/<user>/spikesorting
#   OUTPUT_SCRATCH = <base>/aind_output_scratch
#   DOWNLOAD_BASE  = <base>/aind_output_fordownload
############################################################

# Raw arguments (may be empty when using profile-based paths)
ARG_SCRATCH="${1:-}"
ARG_DOWNLOAD="${2:-}"
USER_PROFILE="${3:-}"

# Default paths (Shun) when no user profile is given
OUTPUT_SCRATCH="${ARG_SCRATCH:-/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_scratch}"
DOWNLOAD_BASE="${ARG_DOWNLOAD:-/n/netscratch/bsabatini_lab/Lab/shunnnli/spikesorting/aind_output_fordownload}"

# If a user profile is provided, override paths using user_profiles.conf
if [ -n "$USER_PROFILE" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    PIPELINE_DIR="$SCRIPT_DIR"
    USER_PROFILES_CONFIG="${PIPELINE_DIR}/user_profiles.conf"

    echo "Using user profile: $USER_PROFILE"
    user_base=""
    if [ -f "$USER_PROFILES_CONFIG" ]; then
        profile_base=$(awk -F= -v p="$USER_PROFILE" '$1==p {print $2}' "$USER_PROFILES_CONFIG" | tail -n 1)
        if [ -n "$profile_base" ]; then
            user_base="$profile_base"
            echo "  Loaded base path from user_profiles.conf: $user_base"
        else
            echo "  Profile '$USER_PROFILE' not found in user_profiles.conf; using default pattern."
        fi
    else
        echo "  No user_profiles.conf found at: $USER_PROFILES_CONFIG"
        echo "  Using default pattern for user base."
    fi

    if [ -z "$user_base" ]; then
        user_base="/n/netscratch/bsabatini_lab/Lab/${USER_PROFILE}/spikesorting"
    fi

    user_base="${user_base%/}"
    OUTPUT_SCRATCH="${user_base}/aind_output_scratch"
    DOWNLOAD_BASE="${user_base}/aind_output_fordownload"
fi

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

