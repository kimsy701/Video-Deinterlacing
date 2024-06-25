#gt이미지들에서 0,2,4번째 이미지의 pixel even층들은 그대로, pixel odd층들만 random한 숫자로 만듦 / 1,3,5번째 이미지의  pixel even층들은 그대로, pixel odd층들만 random한 숫자로 만듦
import os
import numpy as np
from PIL import Image
import random

def load_images_from_folders(root_folder):
    image_list = []
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    for subfolder in subfolders:
        images = []
        for img_file in os.listdir(subfolder):
            if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(subfolder, img_file)
                image = Image.open(img_path)
                print(np.array(image).shape)
                images.append(image)
        image_list.append(images)
    return image_list # [전체 폴더 [각각 sub foler [각각6개 이미지]] ]

def process_images(image_list):
    processed_images = []
    
    for folder_images in image_list:
        processed_folder_images = []
        for idx, img in enumerate(folder_images):
            img_array = np.array(img)
            random_numbers = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
            
            if idx % 2 == 0:
                # 짝수 인덱스의 이미지
                img_array[::2] = img_array[::2]  # 짝수 height 층 유지
                img_array[1::2] = random_numbers[1::2]  # 홀수 height 층 랜덤 값
            else:
                # 홀수 인덱스의 이미지
                img_array[1::2] = img_array[1::2]  # 홀수 height 층 유지
                img_array[::2] = random_numbers[::2]  # 짝수 height 층 랜덤 값
            
            processed_folder_images.append(Image.fromarray(img_array))
        processed_images.append(processed_folder_images)
    
    return processed_images

# 이미지가 저장된 루트 폴더 경로 지정
root_folder = '/home/kimsy701/deinter_venv/train_val_100'  # 루트 폴더 경로를 여기에 지정
