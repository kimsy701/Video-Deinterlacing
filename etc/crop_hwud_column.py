import cv2
import numpy as np
import os
from tqdm import tqdm

def crop_and_pad_image(image_path, output_path, x1, x2, y1, y2, target_width, target_height):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # 원본 이미지의 크기
    h, w = image.shape[:2]

    # 이미지 크롭
    cropped_image = image[y1:y2, x1:x2]
    
    #print("cropped_image.shape",cropped_image.shape) #(2058, 3050, 3)

    # 크롭된 이미지의 새로운 크기
    cropped_h, cropped_w = cropped_image.shape[:2]

    # 패딩 크기 계산
    pad_left = x1 #max(0, (target_width - cropped_w) // 2) #(3240-2058)//2 = 591 -> 95부터 표시
    pad_right = target_width-x2 #max(0, target_width - cropped_w - pad_left)
    pad_top =  y1 #max(0, (target_height - cropped_h) // 2) #(2127 - (2127-69))//2 = 64
    pad_bottom = target_height-y2 # 59 : 2127까지 표시(2187-n-1)  #max(0, target_height - cropped_h - pad_top) #2187-(2127-69)-64 = 2187-2058-64=65 -> 2121 까지 이미지 표시 (2187-65-1)

    # 패딩 추가
    padded_image = cv2.copyMakeBorder(
        cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # 출력 이미지 저장
    cv2.imwrite(output_path, padded_image)

def process_images(input_folder, output_folder, x1, x2, y1, y2, target_width, target_height):
    # 출력 폴더 생성 (존재할 경우 오류 발생하지 않음)
    os.makedirs(output_folder, exist_ok=True)
    
    # 입력 폴더의 모든 이미지 파일을 처리
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.tiff', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        crop_and_pad_image(input_path, output_path, x1, x2, y1, y2, target_width, target_height)

# 설정 값
input_folder = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_mean_weighted_0.1_0.8_0.1_diffweight_finetune_x4.5factor_FULL_DEG_model_005_midonly'
output_folder = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_mean_weighted_0.1_0.8_0.1_diffweight_finetune_x4.5factor_FULL_DEG_model_005_midonly_crop'

x1 = 94
x2 = 3143+1
y1 = 69
y2 = 2127+1
target_width = 3240
target_height = 2187

# 이미지 처리
process_images(input_folder, output_folder, x1, x2, y1, y2, target_width, target_height)
