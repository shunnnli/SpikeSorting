#!/bin/bash

# Run on Kempner cluster with this command:
# pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm

# Input argument - Slurm job description file
Slurm_file=$1

if [ -z "$Slurm_file" ]; then
  echo "Usage: $0 <slurm_file>"
  exit 1
fi

echo "Generating job submission files from the Slurm file: $Slurm_file"

# Extract relevant paths from the Slurm file
top_dir=$(grep "^DATA_PATH" "$Slurm_file" | sed "s/DATA_PATH=//g" | tr -d '"')
work_dir=$(grep "^WORK_DIR" "$Slurm_file" | sed "s/WORK_DIR=//g" | tr -d '"')
out_dir=$(grep "^RESULTS_PATH" "$Slurm_file" | sed "s/RESULTS_PATH=//g" | tr -d '"')

# Extract PIPELINE_PATH (expected to be "./" from your file)
pipeline_path=$(grep "^PIPELINE_PATH" "$Slurm_file" | sed "s/PIPELINE_PATH=//g" | tr -d '"')
# Resolve relative PIPELINE_PATH to an absolute path (based on current directory)
pipeline_code_dir=$(realpath "$pipeline_path")

echo "Top directory: $top_dir"
echo "Pipeline code path: $pipeline_code_dir"

# Define where to save the pipeline job folders
pipeline_save_path="${pipeline_code_dir%/}/pipeline_saved"
mkdir -p "$pipeline_save_path"

# Array to track submitted SLURM job IDs
declare -a job_ids=()

# Get all immediate subdirectories under DATA_PATH
dir_data_array=( $(find "$top_dir" -mindepth 1 -maxdepth 1 -type d) )
dir_array_length=${#dir_data_array[@]}

echo "Found $dir_array_length data directories"
echo "Generating $dir_array_length pipelines"

for element in "${dir_data_array[@]}"
do
    echo "Processing directory: $element"
    folder_name=$(basename "$element")

    # Check if this recording session already has spike-sorted output.
    # We assume that if the session-specific RESULTS_PATH folder has a
    # 'spikesorted' subdirectory, spike sorting has already completed.
    existing_results_folder="${out_dir%/}/${folder_name}_output"
    if [ -d "${existing_results_folder}/spikesorted" ]; then
        echo "  Detected existing spike-sorted output at ${existing_results_folder}/spikesorted"
        echo "  Skipping spike-sorting job for ${folder_name}; post-processing will still run."
        echo ""
        continue
    fi

    new_data_path="DATA_PATH=\"${element}\""
    new_results_path="RESULTS_PATH=\"${out_dir%/}/${folder_name}_output\""
    new_pipeline_path="PIPELINE_PATH=\"${pipeline_code_dir}\""

    # Create job folder inside pipeline_save_path
    job_folder="${pipeline_save_path}/pipeline_${folder_name}"
    job_slurm_script="spike_sort.slrm.${folder_name}"

    mkdir -p "$job_folder"
    cd "$job_folder"

    # Copy the original Slurm file from the pipeline code directory
    cp "$pipeline_code_dir/$Slurm_file" 1.tmp.slrm
    sed "s|^RESULTS_PATH=.*|$new_results_path|g" 1.tmp.slrm > 2.tmp.slrm
    sed "s|^DATA_PATH=.*|$new_data_path|g" 2.tmp.slrm > 3.tmp.slrm
    sed "s|^PIPELINE_PATH=.*|$new_pipeline_path|g" 3.tmp.slrm > "$job_slurm_script"
    rm 1.tmp.slrm 2.tmp.slrm 3.tmp.slrm

    echo "Submitting $job_slurm_script"
    echo "  Data path: ${element}"
    echo "  Results path: ${out_dir%/}/${folder_name}_output"
    echo ""

    sbatch_output=$(sbatch "$job_slurm_script")
    if [ $? -ne 0 ]; then
        echo "❌ Failed to submit job: $job_slurm_script"
        echo "sbatch output:"
        echo "$sbatch_output"
        exit 1
    fi

    echo "$sbatch_output"
    job_id=$(echo "$sbatch_output" | awk '{print $4}')
    if [[ -n "$job_id" ]]; then
        job_ids+=("$job_id")
    else
        echo "⚠️  Could not parse job ID from sbatch output; post-processing wait may not track this job."
    fi

    # Return to the pipeline code directory before processing the next folder
    cd "$pipeline_code_dir"
done
echo "All spike-sorting jobs have been submitted."

# If we captured any job IDs, wait for them to complete before running post-processing
if [ "${#job_ids[@]}" -gt 0 ]; then
    echo "Waiting for ${#job_ids[@]} spike-sorting jobs to finish before running post-processing..."
    while true; do
        all_done=true
        for jid in "${job_ids[@]}"; do
            # If the job is still present in the queue, keep waiting
            if squeue -h -j "$jid" 2>/dev/null | grep -q .; then
                all_done=false
                break
            fi
        done

        if $all_done; then
            echo "All spike-sorting jobs appear to have finished."
            break
        fi

        echo "Some jobs are still running or pending; sleeping for 60 seconds..."
        sleep 60
    done
else
    echo "No job IDs were recorded; skipping wait step and running post-processing immediately."
fi

# Run the AIND export / post-processing script once everything is done
postprocess_script_path="${pipeline_code_dir%/}/postprocess/extract_aind_output.py"
if [ ! -f "$postprocess_script_path" ]; then
    echo "❌ Post-processing script not found at: $postprocess_script_path"
    echo "Please verify the location of extract_aind_output.py."
    exit 1
fi

# Ensure extract_aind_output.py sees the directory containing all recording sessions.
# This will make it process *every* recording session under this top-level directory.
export AIND_INPUT_BASE_DIR="$top_dir"
export AIND_OUTPUT_BASE_DIR="$out_dir"

echo "Starting post-processing with: $postprocess_script_path"
echo "  Using AIND_INPUT_BASE_DIR=${AIND_INPUT_BASE_DIR}"
echo "  Using AIND_OUTPUT_BASE_DIR=${AIND_OUTPUT_BASE_DIR}"

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

python3 "$postprocess_script_path"
post_status=$?

if [ $post_status -ne 0 ]; then
    echo "❌ Post-processing script exited with status $post_status"
    exit $post_status
fi

echo "✅ Post-processing completed successfully."