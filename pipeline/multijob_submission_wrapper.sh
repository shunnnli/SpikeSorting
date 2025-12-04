#!/bin/bash

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

# Get all immediate subdirectories under DATA_PATH
dir_data_array=( $(find "$top_dir" -mindepth 1 -maxdepth 1 -type d) )
dir_array_length=${#dir_data_array[@]}

echo "Found $dir_array_length data directories"
echo "Generating $dir_array_length pipelines"

for element in "${dir_data_array[@]}"
do
    echo "Processing directory: $element"
    folder_name=$(basename "$element")

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

    if ! sbatch "$job_slurm_script"; then
        echo "❌ Failed to submit job: $job_slurm_script"
        exit 1
    fi

    # Return to the pipeline code directory before processing the next folder
    cd "$pipeline_code_dir"
done
