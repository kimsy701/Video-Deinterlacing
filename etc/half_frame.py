######################## 만들어진 60개에서 30개씩만 가져와서 영상 말기 (1,3,5,만 가져오기    &     2,4,6만 가져오기 )  ########################
import os
import shutil
# List and sort files in the input directory
input_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_nobicubic_rst_fi'
files = sorted(os.listdir(input_folder_path))

#### try 1

output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_nobicubic_rst_fi_only30_1'
os.makedirs(output_folder_path, exist_ok=True)

# Enumerate through the files and copy every second one
i=1
for file_idx, filename in enumerate(files):
    ori_file_path = os.path.join(input_folder_path, filename)
    if file_idx % 2 == 0: 
        dest_file_path = os.path.join(output_folder_path, f"{i:08d}.png")
        shutil.copyfile(ori_file_path, dest_file_path)
        i+=1

# #### try 2

output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_nobicubic_rst_fi_only30_2'
os.makedirs(output_folder_path, exist_ok=True)

i=1
for file_idx, filename in enumerate(files):
    ori_file_path = os.path.join(input_folder_path, filename)
    if file_idx % 2 == 1: 
        dest_file_path = os.path.join(output_folder_path, f"{i:08d}.png")
        shutil.copyfile(ori_file_path, dest_file_path)
        i+=1
        
