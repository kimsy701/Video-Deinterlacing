
import os
import shutil

#### 1. rename and put the folders in one folder ####
src_path1="/mnt/sdb/4kdata_filter"
dest_path1="/mnt/sdb/4kdata_filter_fi"

for folders in os.listdir(src_path1):
    folder_path = os.path.join(src_path1, folders)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        fi_folder_name = folders +"_"+ subfolder
        dest_folder_path = os.path.join(dest_path1, fi_folder_name)
        # os.mkdir(dest_folder_path)
        shutil.copytree(subfolder_path, dest_folder_path)
        
        
#### 2. split train and gt ####

#f1 : gt
#f2 : train set
src_path2 = "/mnt/sdb/4kdata_filter_fi"
train_data_path = "/mnt/sdb/deinter_dataset/train_data"
gt_data_path = "/mnt/sdb/deinter_dataset/gt_data"

# Move folders starting with "4kdata_f1_filter" to train_data
for folder in os.listdir(src_path2):
    if folder.startswith("4kdata_f2_filter"):
        src_folder_path = os.path.join(src_path2, folder)
        dest_folder_path1 = os.path.join(train_data_path, folder)
        
        shutil.copytree(src_folder_path, dest_folder_path1)
        
        
        for files in os.listdir(src_folder_path):
            
            src_file = os.path.join(src_folder_path, files)
            dest_file1 = os.path.join(dest_folder_path1, files)
        
            shutil.copyfile(src_file, dest_file1)

# Move folders starting with "4kdata_f2_filter" to gt_data
for folder in os.listdir(src_path2):
    if folder.startswith("4kdata_f1_filter"):
        src_folder_path = os.path.join(src_path2, folder)
        dest_folder_path2 = os.path.join(gt_data_path, folder)
        
        shutil.copytree(src_folder_path, dest_folder_path2)
        
        for files in os.listdir(src_folder_path):
            
            src_file = os.path.join(src_folder_path, files)
            dest_file2 = os.path.join(dest_folder_path2, files)
        
            shutil.copytree(src_file, dest_file2)
