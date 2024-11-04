#!/bin/bash

# Define the list of directory names
dir_names=(
    "240829_01/A001C001_22011051_CANON"
    "240829_02/A002C002_2201108Z_CANON"
    "240829_03/A003C003_220110PL_CANON"
    "240829_04/A004C004_220110KB_CANON"
    "240829_04/A004C005_220110MQ_CANON"
    "240829_05/A005C006_220110YW_CANON"
    "240829_06/A006C007_220110MS_CANON"
    "240829_07/A007C008_220110VQ_CANON"
    "240829_07/A007C009_22011005_CANON"
    "240829_07/A007C010_220110QR_CANON"
    "240829_08/A008C011_220110T4_CANON"
    "240829_08/A008C012_2201108L_CANON"
    "240829_08/A008C013_220110IL_CANON"
    "240829_09/A009C014_220110CO_CANON"
    "240829_10/A010C015_220110FX_CANON"
)

output_names=(
    "A001C001_22011051"
    "A002C002_2201108Z"
    "A003C003_220110PL"
    "A004C004_220110KB"
    "A004C005_220110MQ"
    "A005C006_220110YW"
    "A006C007_220110MS"
    "A007C008_220110VQ"
    "A007C009_22011005"
    "A007C010_220110QR"
    "A008C011_220110T4"
    "A008C012_2201108L"
    "A008C013_220110IL"
    "A009C014_220110CO"
    "A010C015_220110FX" 
)

# Base paths
base_input_path="/media/inshorts/T5 EVO/ren/240829"
base_output_path="/mnt/sde/8월촬영본/tiff_dataset_0829"

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
