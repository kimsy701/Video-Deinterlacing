# dir_name=$(printf "A001_C001_0527Q9_001")
# mkdir -p "/mnt/sdd/6월촬영본/4k/${dir_name}"
# python video_sq_6월.py "/mnt/nas3/8K 촬영본/8k_org/A001_0527N0.RDM/A001_C001_0527Q9.RDC/${dir_name}.mov" "/mnt/sdd/6월촬영본/4k/${dir_name}"

#dir_name=$(printf "A003_C002_0527WV")
dir_names=(
"A003_C002_0527WV"
"A003_C003_0527IL"
"A003_C004_052761"
"A003_C005_052754"
"A003_C006_052701"
"A003_C007_052744"
"A003_C008_05272I"
"A003_C009_0527IY"
"A003_C010_0527D8"
"A003_C011_0527WX"
)

#mkdir -p "/mnt/sdd/6월촬영본/8k_tiff/${dir_name}"
#python video_seq_6월_8k_tiff.py "/mnt/nas3/8K 촬영본/8k_ren/A003_0527FJ.RDM/${dir_name}.mov" "/mnt/sdd/6월촬영본/8k_tiff/${dir_name}"


# Base paths
base_input_path="/mnt/nas3/8K 촬영본/8k_ren"
base_output_path="/mnt/sdd/6월촬영본/8k_tiff"

# Iterate over each directory name in the list
for dir_name in "${dir_names[@]}"; do
    # Create the target directory
    mkdir -p "${base_output_path}/${dir_name}_001"

    # Define the input and output file paths
    input_file="${base_input_path}/A003_0527FJ.RDM/${dir_name}.mov"
    output_file="${base_output_path}/${dir_name}_001"

    # Call the Python script with the input and output paths
    python video_seq_6월_8k_tiff.py "${input_file}" "${output_file}"
done
