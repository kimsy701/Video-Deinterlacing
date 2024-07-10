# -*- coding: utf-8 -*-
# gt이미지들에서 0,2,4번째 이미지의 pixel even층들은 그대로, pixel odd층들만 t+1 frame이미지로 만듦 , 1,3,5번째 이미지의  pixel even층들은 그대로, pixel odd층들만t+1 frame이미지로 만듦


import os
import numpy as np
from PIL import Image
import random
import shutil

def extract_number1(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        second_part = parts[1]
        second_part_num_part = second_part[-3:]  # Take the last 3 characters of the second part
        
        try:
            return int(second_part_num_part)
        except ValueError:
            try:
                return float(second_part_num_part)
            except ValueError:
                return float('inf')  # Return infinity for non-numeric parts
        
    return float('inf')  # Return infinity if no underscore is found


def extract_number2(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        second_part = parts[1]
        
        try:
            return int(second_part)
        except ValueError:
            try:
                return float(second_part_num_part)
            except ValueError:
                return float('inf')  # Return infinity for non-numeric parts
        
    return float('inf')  # Return infinity if no underscore is found

def extract_number3(directory):

    try:
        # Splitting the filename by '.' to separate the numeric part from the extension
        parts = directory.split('.')
        if len(parts) > 1:
            numeric_part = parts[0]  # Take the first part before the dot
            # Convert the numeric part to integer
            return int(numeric_part)
        else:
            return float('inf')  # Return infinity if no extension found (optional)
    except ValueError:
        return float('inf')  # Return infinity if conversion fails
    
    

def slice_to_scene2(src_path, dest_path):
    
    images = sorted(os.listdir(src_path))
    
    for i, img in enumerate(images):
        scene_number = i // 6
        new_folder_name = f"{scene_number}"
        dest_folder_path = os.path.join(dest_path, new_folder_name)
        
        # Ensure the destination directory exists
        os.makedirs(dest_folder_path, exist_ok=True)
        
        src_img_path = os.path.join(src_path, img)
        dest_img_path = os.path.join(dest_folder_path, img)
        print("src_img_path:", src_img_path)
        print("dest_img_path:", dest_img_path)
        shutil.copy(src_img_path, dest_img_path)
            

def load_images_from_folders(root_folder):
    image_list = []

    for subfolder in sorted(os.listdir(root_folder), key=extract_number2):
        images = []
        subfolder_path = os.path.join(root_folder, subfolder)
        for img_file in sorted(os.listdir(subfolder_path), key=extract_number3):
            if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(subfolder_path, img_file)
                image = Image.open(img_path)
                images.append(image)
        image_list.append(images)
    return image_list # [전체 폴더 [각각 sub foler [각각6개 이미지]] ]


def load_images_from_folders2(root_folder):
    image_list = []
    # img_file_li = [f.path for f in sorted(os.scandir(root_folder))]

    for img_file in sorted(os.listdir(root_folder), key=extract_number3):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(root_folder, img_file)
            image = Image.open(img_path)
            # print(np.array(image).shape)
        image_list.append(np.array(image))
    return image_list # [전체 폴더 [각각 sub foler [각각6개 이미지]] ]

def process_images(save_path, image_list):
    processed_images = []
    
    for folder_idx, folder_images in sorted(enumerate(image_list)): #6개의 folder images들 수행
        processed_folder_images = []
        for img_idx, img in sorted(enumerate(folder_images)): 
            img_array = np.array(img)
            print(img.shape) #(448, 3)
            # random_numbers = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
            if img_idx < len(folder_images) - 1:
                next_img_array = np.array(folder_images[img_idx + 1])
            else:
                next_img_array = np.array(folder_images[img_idx-1])  #마지막 frame이면, 다음 프레임 이미지 대신 이전 프레임 이미지 넣기
            
            if img_idx % 2 == 0:
                # 짝수 인덱스의 이미지
                img_array[1::2] = img_array[1::2]  # 짝수 height 층 유지
                # img_array[1::2] = random_numbers[1::2]  # 홀수 height 층 랜덤 값
                img_array[::2] = next_img_array[::2]
            else:
                # 홀수 인덱스의 이미지
                img_array[::2] = img_array[::2]  # 홀수 height 층 유지
                # img_array[::2] = random_numbers[::2]  # 짝수 height 층 랜덤 값
                img_array[1::2] = next_img_array[1::2] 
            
            processed_folder_images.append(Image.fromarray(img_array))
            
        # Save processed images
        output_folder = os.path.join(save_path, f'{folder_idx}')
        os.makedirs(output_folder, exist_ok=True)
        for img_idx2, img2 in enumerate(processed_folder_images):
            img_path=os.path.join(output_folder, f'{img_idx2:08d}.png')
            img2.save(img_path)



def process_images2(save_path, image_list):
    # processed_images = []
    for img_idx, img in sorted(enumerate(image_list)): #6개의 folder images들 수행

        img_array = np.array(img)
        # random_numbers = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
        if img_idx < len(image_list) - 1:
            next_img_array = np.array(image_list[img_idx + 1])
        else:
            next_img_array = np.array(image_list[img_idx-1])  #마지막 frame이면, 다음 프레임 이미지 대신 이전 프레임 이미지 넣기
        
        if img_idx % 2 == 0:
            # 짝수 인덱스의 이미지
            img_array[1::2] = img_array[1::2]  # 짝수 height 층 유지
            # img_array[1::2] = random_numbers[1::2]  # 홀수 height 층 랜덤 값
            img_array[::2] = next_img_array[::2]
        else:
            # 홀수 인덱스의 이미지
            img_array[::2] = img_array[::2]  # 홀수 height 층 유지
            # img_array[::2] = random_numbers[::2]  # 짝수 height 층 랜덤 값
            img_array[1::2] = next_img_array[1::2] 
        
        # processed_images.append(Image.fromarray(img_array))
        
        # Save processed images
        img_path=os.path.join(save_path, f'{img_idx:08d}.png')
        Image.fromarray(img_array).save(img_path)


#### Process ####
"""
for i in range(47):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C002_052717_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C002_052717_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)
"""

for i in range(434):
    # root_path="/mnt/sdc/4K_실사/1.바비/PNG_dataset/scene_000300"
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C003_0527HZ_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C003_0527HZ_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)

for i in range(496):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C004_0527KN_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C004_0527KN_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)

for i in range(434):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C005_0527MR_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C005_0527MR_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)

for i in range(435):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C006_0527WR_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C006_0527WR_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)

for i in range(419):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C007_05278O_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C007_05278O_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)

for i in range(290):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C010_0527P3_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C010_0527P3_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)

for i in range(650):
    scened_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel/A001_C011_0527OW_001/scene_{i}"
    save_path=f"/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_interlace/A001_C011_0527OW_001/scene_{i}"
    os.makedirs(save_path, exist_ok=True)
    # slice_to_scene2(root_path, scened_path)
    img_list1=load_images_from_folders2(scened_path)
    process_images2(save_path, img_list1)
