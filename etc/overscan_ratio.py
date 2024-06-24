################################################   데이터가 overscan되어있고, 비율이 맞지 않는 다면, 이를 처리하는 방법   ################################################ 


### overscan 처리   -  720:486 -> 오버스캔 처리시 704:480 
from PIL import Image
import numpy as np
import os 
import shutil
import re

def process_overscaned(deitner_rst_path, dest_path):
    for subfolder in sorted(os.listdir(deitner_rst_path)):
        subfolder_path =  os.path.join(deitner_rst_path,subfolder)
        #1. deinter output으로 overscan 처리
        for files in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(deitner_rst_path,subfolder_path, files)
            image = Image.open(file_path)
            # Convert the image to a numpy array
            image_array = np.array(image)

            # Calculate the cropping dimensions
            # crop_height = 3
            crop_height = 0 #이미 deinter 하기 전에 자름 
            crop_width = 8
            
            # Crop the image
            cropped_image_array = image_array[:, crop_width:-crop_width]
                
            # Convert the cropped array back to an image
            cropped_image = Image.fromarray(cropped_image_array)
            
            # Save the cropped image
            output_image_path = os.path.join(dest_path,f'{subfolder}_{files}')
            cropped_image.save(output_image_path,compress_level=0)

        print(f"Cropped image saved to {output_image_path}")
    
###### 함수 실행 코드 ######
# deitner_rst_path="/home/kimsy701/deinter_dataset/train_val_winter_fi"
# dest_path="/home/kimsy701/overscan_ratio/overscan_deinter_ratio/overscan"
# process_overscaned(deitner_rst_path, dest_path)

#for sudo gt
# deitner_rst_path="/home/kimsy701/deinter_dataset/gt_val_winter_fi"
# dest_path="/home/kimsy701/overscan_ratio/overscan_deinter_ratio/sudo_gt_overscan"
# process_overscaned(deitner_rst_path, dest_path)




###이미지 명 처리 ###
def extract_number(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        first_part =parts[0] #parts[-1]
        # print("first_part",first_part )
        # second_part = os.path.splitext(parts[-1])[0] 
        second_part = os.path.splitext(parts[-1])[0][-1] 
        # print("second_part",second_part )
        try:
            return int(first_part)*100000 +int(second_part)
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
        
###### 함수 실행 코드 ######

# src_path = "/home/kimsy701/overscan_ratio/overscan_deinter_ratio/overscan"
# dest_path = "/home/kimsy701/overscan_ratio/overscan_deinter_ratio/overscan_fi"
# # os.mkdir(dest_path)
# rename_files(src_path,dest_path)




### png to mp4 

### 가로 픽셀 비율 처리 (ffmpeg)   -   오버스캔 처리시 704:480 -> 픽셀비율 조정시 640:480
#  ffmpeg -i ./*.png -vf "scale=640:480,setsar=1" -c:a copy output/%08d.png
# ffmpeg -y -i /home/kimsy701/overscan/after_deinter/after_overscan/*.png -vf "scale=640:480,setsar=1" -c:a copy /home/kimsy701/overscan/after_deinter/after_resizeratio/%08d.png
# ffmpeg -y -i '/home/kimsy701/overscan/after_deinter/after_overscan/%08d.png' -vf "scale=640:480,setsar=1" -c:a copy '/home/kimsy701/overscan/after_deinter/after_resizeratio/%08d.png'
#ffmpeg -y -i '/home/kimsy701/overscan/overscan_ratio_deinter/after_overscan_fi/%08d.png' -vf "scale=640:240,setsar=1" -c:a copy '/home/kimsy701/overscan/overscan_ratio_deinter/after_ratio/%08d.png'

"""
-vf "scale=iw*par:ih,setsar=1" is a video filter (-vf) that:
    scale=iw*par:ih scales the video according to its pixel aspect ratio.
    setsar=1 sets the storage aspect ratio to 1 (square pixels).
-c:a copy copies the audio stream without re-encoding.
"""




###### 순서대로 6개씩 폴더 생성하여 넣기 ######
import os
import shutil

def extract_number2(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def copy_files_in_batches(src_folder, dest_folder, batch_size=6):
    # files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')], key=extract_number2)
    files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
    
    batch_num = 1
    for i in range(0, len(files), batch_size):
        batch_folder = os.path.join(dest_folder, str(batch_num))
        os.makedirs(batch_folder, exist_ok=True)
        
        for file in files[i:i + batch_size]:
            src_file_path = os.path.join(src_folder, file)
            dest_file_path = os.path.join(batch_folder, file)
            shutil.copy(src_file_path, dest_file_path)
        
        batch_num += 1

# Example usage
# src_folder = '/home/kimsy701/overscan_ratio/overscan_deinter_ratio/overscan_fi'
# dest_folder = '/home/kimsy701/overscan_ratio/overscan_deinter_ratio/deinter_input'
# src_folder = '/home/kimsy701/overscan_ratio/overscan_deinter_ratio/sudo_gt_overscan_fi'
# dest_folder = '/home/kimsy701/overscan_ratio/overscan_deinter_ratio/sudo_gt_overscan_fi_fi'
# # os.mkdir(dest_folder)
# copy_files_in_batches(src_folder, dest_folder)


### 6개씩 묶여져 있다면 그걸 하나의 폴더에 다 넣자 ###
def extract_number3(directory):
    parts = directory.split('.')
    if len(parts) > 1:
        first_part =parts[0] #parts[-1]

        try:
            return int(first_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)

def copy_and_rename_files(src_folders, dest_folder):
    for subfolders in sorted(os.listdir(src_folders), key = extract_number3):
        subfolder_path = os.path.join(src_folders, subfolders)
        print(subfolder_path)
        for imgs in sorted(os.listdir(subfolder_path), key = extract_number3):
            img_path =os.path.join(subfolder_path,imgs)
            print("img_path",img_path)
            dest_img_path =os.path.join(dest_folder,f'{subfolders}_{imgs}')
            shutil.copy(img_path, dest_img_path)
            
# src_folder = '/home/kimsy701/overscan_ratio/overscan_deinter_ratio/deinter_output'
# dest_folder = '/home/kimsy701/overscan_ratio/overscan_deinter_ratio/deinter_output_fi'
# os.mkdir(dest_folder)
# copy_and_rename_files(src_folder, dest_folder)


#rename files again..
# src_path = "/home/kimsy701/overscan_ratio/overscan_deinter_ratio/deinter_output_fi"
# dest_path = "/home/kimsy701/overscan_ratio/overscan_deinter_ratio/deinter_output_fi_fi"
# os.mkdir(dest_path)
# rename_files(src_path,dest_path)
