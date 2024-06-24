import os
import shutil
import re

# List of folder names


# Function to extract sorting keys
def extract_key(folder_name):
    # Extract numerical ranges and suffixes
    match = re.match(r'4kdata_f1_filter_(\d+)-(\d+)_(\d+)_([A-Z]?)', folder_name)
    if match:
        start, end, subfolder, suffix = match.groups()
        # Convert extracted strings to integers where applicable
        return (int(start), int(subfolder), suffix)
    return folder_name


#rename folder names
src_path = "/mnt/sdb/deinter/deinter_dataset/gt_data"
dest_path = "/mnt/sdb/deinter/deinter_dataset/gt_data_fi"

for i, subfolder in enumerate(sorted(os.listdir(src_path),key=extract_key)):
    print(subfolder)
    new_foldername = i
    
    shutil.copytree(os.path.join(src_path,subfolder), os.path.join(dest_path, str(i)))
    
