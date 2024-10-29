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

def alphabet_to_number(letter):
    return ord(letter.upper()) - 64

def extract_number4(directory):
    parts = directory.split('_')
    
    part0 = parts[0][-2:]
    part1 = parts[1] #01
    part2 = alphabet_to_number(parts[2]) #A -> 1, B-> 2,...
    part3 = parts[3] #0000
    part4 = os.path.splitext(parts[4])[0][-1] #im1, im3 의 1,3 부분
        
    return int(part0)*1000000 + int(part1)*10000 + int(part2)*1000 + int(part3)*100 + int(part4)

def extract_number5(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        first_part =parts[0]
        # print("first_part",first_part )

        second_part = os.path.splitext(parts[-1])[0]
        # print("second_part",second_part )
        try:
            return int(first_part)*100000 +int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)


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

def rename(src_path2, des_path2):
    os.makedirs(dest_path2, exist_ok=True)
    fi_file_name=1
    for idx, files in enumerate(sorted(os.listdir(src_path2), key=extract_number3)):   #change the key, according to the scr path's folder name pattern 
        print("os.path.splitext(files)[0]",os.path.splitext(files)[0])
        # file_name_int = int(os.path.splitext(files)[0])
        
        ori_file_path = os.path.join(src_path2, files)
        new_name =f'{fi_file_name:08d}.tiff'
        dest_folder_path = os.path.join(dest_path2, new_name)
            # os.mkdir(dest_folder_path)
        shutil.copy(ori_file_path, dest_folder_path)
        fi_file_name+=1
      
src_path2="/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice_bicubic"
dest_path2="/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice_bicubic_rename"
# src_path2="/mnt/sdb/VSR_Inference/sr_test_samples/littlewomen2_multidataset_intraattn_hflf10_decx5_less_deg_gloss_scratch_x2_1324"
# dest_path2="/mnt/sdb/VSR_Inference/sr_test_samples/littlewomen2_multidataset_intraattn_hflf10_decx5_less_deg_gloss_scratch_x2_1324_rename" 
rename(src_path2, dest_path2)
