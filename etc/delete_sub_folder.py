import os
import shutil

# Define the path to the main directory
main_directory = '/home/kimsy701/gt_val_re'

# Iterate over each folder in the main directory
for main_folder in os.listdir(main_directory):
    main_folder_path = os.path.join(main_directory, main_folder)
    
    # Ensure we are only processing directories
    if os.path.isdir(main_folder_path):
        # List subdirectories within the current main folder
        for subfolder in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder)
            
            # Ensure we are only processing directories
            if os.path.isdir(subfolder_path):
                try:
                    # Attempt to remove the subdirectory
                    os.rmdir(subfolder_path)
                    print(f'Successfully removed empty subdirectory: {subfolder_path}')
                except OSError:
                    # If the subdirectory is not empty, use shutil.rmtree to remove it
                    shutil.rmtree(subfolder_path)
                    print(f'Successfully removed non-empty subdirectory: {subfolder_path}')
