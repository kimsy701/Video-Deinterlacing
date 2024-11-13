#!/bin/bash

# Define the list of directory names
dir_names=(
    "8월촬영본_1_final_vapoursynthbobqtgmc"
)

output_names=(
"0"
)

# Base paths
base_input_path="/mnt/sda/qtgmc_dataset"
base_output_path="/mnt/sda/qtgmc_dataset"

# Iterate over each directory name in the list
for idx in $(seq 0 $((${#dir_names[@]} - 1))); do
    # Create the target directory
    mkdir -p "${base_output_path}/${output_names[idx]}"

    # Define the input and output file paths
    input_file="${base_input_path}/${dir_names[idx]}.mov"
    output_file="${base_output_path}/${output_names[idx]}"

    # Call the Python script with the input and output paths
    python3 video_seq_8월_4k_tiff.py "${input_file}" "${output_file}"
done
