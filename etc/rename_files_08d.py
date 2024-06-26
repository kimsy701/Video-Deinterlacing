import os
import shutil


def extract_number(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        last_part =os.path.splitext(parts[-1])[0] 
        # print("first_part",first_part )

        second_last_part = parts[-2].split("rst")[-1]
        # print("second_part",second_part )
        try:
            return int(second_last_part)*100000 +int(last_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)


def extract_number2(directory):
    return int(directory)

def extract_number3(directory):
    parts = os.path.splitext(directory)[0]
    return int(parts)




#### 1. rename and put the folders in one folder ####

# src_path1="/home/kimsy701/deinter_venv/train_val_winter_searaft_rst"
# dest_path1="/home/kimsy701/deinter_venv/train_val_winter_searaft_rst_rst"

# for folders in sorted(os.listdir(src_path1), key=extract_number2):
#     folder_path = os.path.join(src_path1, folders)
#     for files in sorted(os.listdir(folder_path), key=extract_number3):
#         file_path = os.path.join(folder_path, files)
#         dest_folder_path = os.path.join(dest_path1, f'{folders}_{files}')
#         # os.mkdir(dest_folder_path)
#         shutil.copy(file_path, dest_folder_path)
        
        
        
#### 2. rename files ####

src_path2="/home/kimsy701/deinter_venv/gt_val_winter_fi"
dest_path2="/home/kimsy701/deinter_venv/gt_val_winter_fi_fi"

for idx, files in enumerate(sorted(os.listdir(src_path2), key=extract_number2)):   #change the key, according to the scr path's folder name pattern 
    ori_file_path = os.path.join(src_path2, files)
    new_name =f'{idx+1:08d}.png'
    dest_folder_path = os.path.join(dest_path2, new_name)
        # os.mkdir(dest_folder_path)
    shutil.copy(ori_file_path, dest_folder_path)
