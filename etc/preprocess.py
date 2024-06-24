#from preprocess_class import Slice,Slicedto4SC




#######  !!!!!!!!!!!!!!!!!!!!!!!!!!!     겨울연가 같은 비디오를 처음 처리하는 과정     !!!!!!!!!!!!!!!!!!!!!!!!!!! 


#저장
################################################### 전체 png들을 6개씩 나눠서 각각의 폴더(1,2,...)에 담기 ###################################################
import os
import shutil
from PIL import Image
import numpy as np

def organize_images_by_batches(src_folder, dest_path, batch_size=6):
    # 1. 폴더 내 모든 PNG 파일 목록 가져오기
    png_files = [f for f in os.listdir(src_folder) if f.endswith('.png')]
    png_files.sort()  # 파일 이름 순으로 정렬 (1.png, 2.png, ...)

    # 2. 배치 단위로 새로운 폴더 생성 및 이미지 이동
    for i in range(0, len(png_files), batch_size):
        batch = png_files[i:i + batch_size]
        batch_folder = os.path.join(dest_path, f'{i // batch_size + 1}')
        os.makedirs(batch_folder, exist_ok=True)

        for file_name in batch:
            src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(batch_folder, file_name)
            shutil.copy2(src_file_path, dest_file_path)
            print(f'Moved {src_file_path} to {dest_file_path}')

# 사용 예시
src_folder = '/home/kimsy701/infer_data_ori/겨울연가_1080'

#src_folder
## -png,png,png

#dest_path
##folder1, folder2

dest_path = '/home/kimsy701/deinter_venv/train_val_winter1080'  # 이미지 파일들이 있는 폴더 경로를 입력하세요
organize_images_by_batches(src_folder,dest_path)


################################################### 8의 배수의 크기로 resize ###################################################
# 486 -> 위 아래 3,3 씩 자르기 
# /home/kimsy701/deinter_venv/train_val_MS 에서 /home/kimsy701/deinter_venv/train_val_MS_crop으로 
input_folder_path='/home/kimsy701/deinter_venv/train_val_winter'
output_folder_path='/home/kimsy701/deinter_venv/train_val_winter_crop'

for folder in sorted(os.listdir(input_folder_path)):
    frames = [] #for each 1_1, 1_2 folder
    folder_path = os.path.join(input_folder_path, folder)
    
    for filename in sorted(os.listdir(folder_path)):
      if filename.endswith(('.png', '.jpg', '.jpeg')):
          file_path = os.path.join(folder_path, filename)
          img = Image.open(file_path)
          frames.append((filename, np.array(img)))  # Store filename along with the frame
          
    for filename, frame in frames:
      # Process for odd and even fields
        processed_frame = frame[3:483,:,:] #3,4,5....482
        
        # Convert the processed frame back to a PIL Image
        processed_img = Image.fromarray(processed_frame)

        # Construct the output filename including both folder name and original filename
        output_filename = f"{filename}"  # Concatenate folder name and original filename
        output_path = os.path.join(output_folder_path, folder, output_filename)

        if not os.path.exists(os.path.join(output_folder_path, folder)):
            os.makedirs(os.path.join(output_folder_path, folder))

        processed_img.save(output_path, compress_level=0)
        # processed_img.save(output_path, compress_level=None)

################################################### (slice): 6개 단위로 1,3,5는 even만, 2,4,6은 odd만 가져오기 ###################################################

import os
import numpy as np
from PIL import Image
import torch.nn as nn

class Slice_pred(nn.Module):

    def __init__(self):
        super(Slice_pred, self).__init__()

    def forward(self, image):
        frames_output = []
        # gt_frames_output=[]

        # Process frames for odd and even fields and save them
        for i, img in enumerate(image):
            # Process for odd and even fields
            processed_img = img[1::2, :] if i % 2 == 0 else img[::2, :] #val_ version1
            frames_output.append(processed_img)

        return frames_output
    


# Define source directory
source_dir = "/home/kimsy701/deinter_venv/train_val_winter_crop"

# Define destination directory
#destination_dir = os.path.join(os.getcwd(), 'train_sliced')
#gt_destination_dir = os.path.join(os.getcwd(), 'gt_sliced')

destination_dir1="/home/kimsy701/deinter_venv/train_val_winter_fi"
# gt_destination_dir="C:\\Users\\인쇼츠\\Desktop\\Deinterlacing구현\\inference_data\\train\\img\\gt_sliced\\21_21"

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir1):
    os.makedirs(destination_dir1)


    
# Iterate over each folder in the source directory
for folder_name in sorted(os.listdir(source_dir)):  #only first 72436 folder
    if os.path.isdir(os.path.join(source_dir, folder_name)):
        # Construct source folder path
        source_folder_path = os.path.join(source_dir, folder_name)
        # Construct destination folder path
        destination_folder_path1 = os.path.join(destination_dir1, folder_name)
        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder_path1):
            os.makedirs(destination_folder_path1)


        # Load images from the source folder
        images = []  # List to hold images as arrays for processing
        for filename in sorted(os.listdir(source_folder_path)):
            if filename.endswith('.png'):  # Assuming images are PNG files
                img = Image.open(os.path.join(source_folder_path, filename))
                images.append(np.array(img))

        # Instantiate Slice model
        slice_model = Slice_pred()
        # Process images using Slice model
        # processed_images, gt_processed_images = slice_model(images)
        processed_images = slice_model(images)

        # Save processed images to the destination folder
        for i, processed_img in enumerate(processed_images):
            processed_img = Image.fromarray(processed_img)
            processed_img.save(os.path.join(destination_folder_path1, f'im{i+1}.png'),compress_level=0)
            


print("Images processed and saved successfully!")

# #저장
