#!/bin/bash

# Define the list of directory names
dir_names=(
    "A001_C002_052717"
    "A001_C003_0527HZ"
    "A001_C004_0527KN"
    "A001_C005_0527MR"
    "A001_C006_0527WR"
    "A001_C007_05278O"
    "A001_C008_0527DM"
    "A001_C009_05277P"
    "A001_C010_0527P3"
    "A001_C011_0527OW"
)

# Base paths
base_input_path="/mnt/nas3/8K 촬영본/8k_org"
base_output_path="/mnt/sdd/6월촬영본/4k_tiff"

# Iterate over each directory name in the list
for dir_name in "${dir_names[@]}"; do
    # Create the target directory
    mkdir -p "${base_output_path}/${dir_name}_001"

    # Define the input and output file paths
    input_file="${base_input_path}/A001_0527N0.RDM/${dir_name}.RDC/${dir_name}_001.mov"
    output_file="${base_output_path}/${dir_name}_001"

    # Call the Python script with the input and output paths
    python video_seq_6월_4k_tiff.py "${input_file}" "${output_file}"
done
