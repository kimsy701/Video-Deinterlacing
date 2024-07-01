import zipfile
import os
import shutil
################ unzip and make it to "21_1" format ################
def strip_leading_zeros(folder_name):
    # Remove leading zeros from the folder name
    return folder_name.lstrip('0')

def move_contents(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.move(s, d)
    os.rmdir(src)  # Remove the now-empty subfolder

def extract_and_rename(zip_file_path, destination_path):
    # Create a temporary directory for extraction
    temp_extract_path = os.path.join(destination_path, 'temp')
    os.makedirs(temp_extract_path, exist_ok=True)

    # Extract the zip file to the temporary directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # Process each main folder in the temporary extraction directory
    main_folders = [f for f in os.listdir(temp_extract_path) if os.path.isdir(os.path.join(temp_extract_path, f))]

    for main_folder in main_folders:
        main_folder_stripped = strip_leading_zeros(main_folder)
        main_folder_path = os.path.join(temp_extract_path, main_folder)
        
        # Get all subfolders in the main folder
        subfolders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
        
        for subfolder in subfolders:
            subfolder_stripped = strip_leading_zeros(subfolder)
            new_folder_name = f'{main_folder_stripped}_{subfolder_stripped}'
            new_folder_path = os.path.join(destination_path, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            old_folder_path = os.path.join(main_folder_path, subfolder)
            
            # Move contents of the subfolder to the new folder
            move_contents(old_folder_path, new_folder_path)

    # Remove the temporary extraction directory
    shutil.rmtree(temp_extract_path)

    print('Extraction and renaming completed successfully.')
    
    
# Define the path to the zip file and the destination directory
zip_file_path = '/home/kimsy701/vimeo_90k/vimeo_90k_10.zip'
destination_path = '/home/kimsy701/test'

extract_and_rename(zip_file_path, destination_path)



################ make 7 frames to 6 frames ################
import os

# Path to the directory containing your subdirectories
base_dir = "/home/kimsy701/gt_val_re"

# Iterate over each subdirectory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    # Check if the item is a directory
    if os.path.isdir(subdir_path):
        # List all files in the subdirectory
        files = os.listdir(subdir_path)
        # Check if 'im7' exists in the list of files
        if 'im7.png' in files:
            # Remove 'im7' from the list of files
            files.remove('im7.png')
            # Remove 'im7' file from the subdirectory
            os.remove(os.path.join(subdir_path, 'im7.png'))
