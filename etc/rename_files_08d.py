# train_val_winter_fi_rst_rst0_0.png -> 1.png 등등으로 index로 이름 rename

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



def rename_files(src_folder, dest_path):
    files = sorted(os.listdir(src_folder), key=extract_number)
    for idx, files in enumerate(files):
        print("files", files)
        file_path = os.path.join(src_folder, files)
        new_name =f'{idx+1:08d}.png'
        dest_file_path = os.path.join(dest_path,new_name)
        shutil.copy(file_path, dest_file_path)
        
        
src_path = "/mnt/sdb/deinter/deinter_dataset/train_val_winter_fi_rst_rst"
dest_path = "/mnt/sdb/deinter/deinter_dataset/train_val_winter_fi_rst_rst_rst"
os.mkdir(dest_path)
rename_files(src_path,dest_path)
