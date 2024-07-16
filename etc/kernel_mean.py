import numpy as np
import cv2
from scipy.ndimage import convolve
import os
import glob

# 이미지 불러오기
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# 결과 이미지 저장
def save_image(image, save_img_path):
    cv2.imwrite(save_img_path, image)

# 커널 평균 적용
def apply_kernel_average(image, kernel_size):
    # 3x3 평균 커널 생성
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # 채널별로 연산하기 위해 이미지 분리
    channels = cv2.split(image)
    avg_channels = []

    for channel in channels:
        # 패딩을 사용하여 에지 처리
        padded_channel = np.pad(channel, ((1, 1), (1, 1)), mode='reflect')
        # 커널과 컨볼루션하여 지역 평균값 계산
        avg_channel = convolve(padded_channel, kernel, mode='constant')[1:-1, 1:-1]
        avg_channels.append(avg_channel)

    # 평균값 채널들을 다시 합치기
    avg_image = cv2.merge(avg_channels)

    # 마스크된 부분을 평균값으로 치환
    replace_image = image.copy()
    replace_image= avg_image

    return replace_image

# 폴더 내의 모든 이미지 처리
def process_images_in_folder(input_folder, output_folder,kernel_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = sorted(glob.glob(os.path.join(input_folder, '*.png')))
    
    for i, image_path in enumerate(image_paths):
        if i>1080:
            print("i",i)
            image = load_image(image_path)
            processed_image = apply_kernel_average(image,kernel_size)
            
            # 결과 이미지 저장
            save_img_path = os.path.join(output_folder, os.path.basename(image_path))
            save_image(processed_image, save_img_path)

# 메인 함수
if __name__ == '__main__':
    kernel_size=5
    input_folder = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst' 
    output_folder = f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_kernel_mean_full_image_size{kernel_size}'

    process_images_in_folder(input_folder, output_folder,kernel_size)
