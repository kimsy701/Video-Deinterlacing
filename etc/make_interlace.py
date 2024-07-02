# -*- coding: utf-8 -*-
# gt이미지들에서 0,2,4번째 이미지의 pixel even층들은 그대로, pixel odd층들만 t+1 frame이미지로 만듦 , 1,3,5번째 이미지의  pixel even층들은 그대로, pixel odd층들만t+1 frame이미지로 만듦


import os
import numpy as np
from PIL import Image
import random
import shutil

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
    subfolders = [f.path for f in sorted(os.scandir(root_folder), key=lambda x: x.name) if f.is_dir()]
    print("subfolders",subfolders)
    for subfolder in subfolders:
        images = []
        for img_file in sorted(os.listdir(subfolder)):
            print("img_file,",img_file)
            if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(subfolder, img_file)
                image = Image.open(img_path)
                print(np.array(image).shape)
                images.append(image)
        image_list.append(images)
    return image_list # [전체 폴더 [각각 sub foler [각각6개 이미지]] ]


def load_images_from_folders2(root_folder):
    image_list = []
    img_file_li = [f.path for f in sorted(os.scandir(root_folder))]

    for img_file in img_file_li:
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(root_folder, img_file)
            image = Image.open(img_path)
            # print(np.array(image).shape)
        image_list.append(np.array(image))
    return image_list # [전체 폴더 [각각 sub foler [각각6개 이미지]] ]

def process_images(save_path, image_list):
    processed_images = []
    
    for folder_idx, folder_images in sorted(enumerate(image_list)): #6개의 folder images들 수행
        print("folder_idx",folder_idx)
        processed_folder_images = []
        for img_idx, img in sorted(enumerate(folder_images)): 
            print("img_idx", img_idx)
            img_array = np.array(img)
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
            print("img_path", img_path)
            img2.save(img_path)



#### Process ####
# root_path="/mnt/sdc/4K_실사/1.바비/PNG_dataset/scene_000300"
scened_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/4k_바비_cropped/4k_바비_cropped_ori"
save_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/4k_바비_cropped/4k_바비_cropped_inter"
# slice_to_scene2(root_path, scened_path)
img_list1=load_images_from_folders(scened_path)
process_images(save_path, img_list1)
