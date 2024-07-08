#두개 시간이 겹친 원본이 들어왔다고 가정함. 
#Assume TFF, Assume BFF  두개의 방식으로 전처리 




#######  !!!!!!!!!!!!!!!!!!!!!!!!!!!     겨울연가 같은 비디오를 처음 처리하는 과정     !!!!!!!!!!!!!!!!!!!!!!!!!!! 


#저장
################################################### 전체 png들을 6개가 아니고 3개씩 나눠서 각각의 폴더(1,2,...)에 담기 ###################################################
import os
import shutil
from PIL import Image
import numpy as np

def organize_images_by_batches(src_folder, dest_path, batch_size=3):
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
src_folder = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/원본PNG'

#src_folder
## -png,png,png

#dest_path
##folder1, folder2

dest_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/gt_val_winter_two_time'  # 이미지 파일들이 있는 폴더 경로를 입력하세요
# organize_images_by_batches(src_folder,dest_path)


################################################### 8의 배수의 크기로 resize ###################################################
# 486 -> 위 아래 3,3 씩 자르기 
# /home/kimsy701/deinter_venv/train_val_MS 에서 /home/kimsy701/deinter_venv/train_val_MS_crop으로 
# input_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/gt_val_winter_two_time'
# output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/gt_val_winter_two_time_fi_pad'

# for folder in sorted(os.listdir(input_folder_path)):
#     frames = [] #for each 1_1, 1_2 folder
#     folder_path = os.path.join(input_folder_path, folder)
    
#     for filename in sorted(os.listdir(folder_path)):
#       if filename.endswith(('.png', '.jpg', '.jpeg')):
#           file_path = os.path.join(folder_path, filename)
#           img = Image.open(file_path)
#           frames.append((filename, np.array(img)))  # Store filename along with the frame
          
#     for filename, frame in frames:
#       # Process for odd and even fields
#         # processed_frame = frame[3:483,:,:] #3,4,5....482
#         padding = ((13, 13), (0, 0), (0, 0))  # (top, bottom), (left, right), (color channels)
#         processed_frame = np.pad(frame, padding, mode='constant', constant_values=0)
        
#         # Convert the processed frame back to a PIL Image
#         processed_img = Image.fromarray(processed_frame)

#         # Construct the output filename including both folder name and original filename
#         output_filename = f"{filename}"  # Concatenate folder name and original filename
#         output_path = os.path.join(output_folder_path, folder, output_filename)

#         if not os.path.exists(os.path.join(output_folder_path, folder)):
#             os.makedirs(os.path.join(output_folder_path, folder))

#         processed_img.save(output_path, compress_level=0)
        



################################################### 3장 -> 6장 만드는 함수 ###################################################
import numpy as np
from PIL import Image

def tff_two_time(input_image_path, t_frame_output_path, t1_frame_output_path):
    """
    Interlaced 이미지를 t 프레임과 t+1 프레임으로 분리하여 저장합니다.

    :param input_image_path: interlaced 이미지 파일 경로
    :param t_frame_output_path: t 프레임 출력 이미지 파일 경로
    :param t1_frame_output_path: t+1 프레임 출력 이미지 파일 경로
    """
    # 이미지를 읽어옵니다.
    image = Image.open(input_image_path)
    image_data = np.array(image)

    # 이미지의 크기를 확인합니다.
    height, width = image_data.shape[:2]
    assert height % 2 == 0, "The height of the image should be even."

    # t 프레임과 t+1 프레임을 저장할 빈 배열을 생성합니다.
    t_frame = np.zeros((height // 2, width, 3), dtype=image_data.dtype)
    t1_frame = np.zeros((height // 2, width, 3), dtype=image_data.dtype)

    # 홀수 행(t 프레임)과 짝수 행(t+1 프레임)을 분리합니다.
    t_frame = image_data[0::2]
    t1_frame = image_data[1::2]

    # 두 프레임을 이미지로 변환합니다.
    t_image = Image.fromarray(t_frame)
    t1_image = Image.fromarray(t1_frame)

    # 분리한 이미지를 저장합니다.
    t_image.save(t_frame_output_path)
    t1_image.save(t1_frame_output_path)

# 예시 사용
# tff_two_time('interlaced_image.png', 't_frame.png', 't1_frame.png')

    

def bff_two_time(input_image_path, t_frame_output_path, t1_frame_output_path):
    """
    Bottom Frame First 방식의 interlaced 이미지를 t 프레임과 t+1 프레임으로 분리하여 저장합니다.

    :param input_image_path: interlaced 이미지 파일 경로
    :param t_frame_output_path: t 프레임 출력 이미지 파일 경로
    :param t1_frame_output_path: t+1 프레임 출력 이미지 파일 경로
    """
    # 이미지를 읽어옵니다.
    image = Image.open(input_image_path)
    image_data = np.array(image)

    # 이미지의 크기를 확인합니다.
    height, width = image_data.shape[:2]
    assert height % 2 == 0, "The height of the image should be even."

    # 홀수 행(t 프레임)과 짝수 행(t+1 프레임)을 분리합니다.
    t1_frame = image_data[0::2]
    t_frame = image_data[1::2]

    # 두 프레임을 이미지로 변환합니다.
    t_image = Image.fromarray(t_frame)
    t1_image = Image.fromarray(t1_frame)

    # 분리한 이미지를 저장합니다.
    t_image.save(t_frame_output_path)
    t1_image.save(t1_frame_output_path)


################################################### 3장 -> 6장 만드는 함수 수행하기  ###################################################

def process_folders_tff(ori_path, des_path):
    """
    ori_path 안의 subfolder들을 탐색하여 각 subfolder 안의 이미지를 처리하고,
    des_path에 새로운 subfolder를 만들어서 결과 이미지를 저장합니다.

    :param ori_path: 원본 폴더 경로
    :param des_path: 결과를 저장할 폴더 경로
    """
    # 원본 폴더 내의 모든 subfolder를 탐색합니다.
    for subfolder_name in sorted(os.listdir(ori_path)):
        subfolder_path = os.path.join(ori_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            # 결과 폴더를 생성합니다.
            dest_subfolder_path = os.path.join(des_path, subfolder_name)
            os.makedirs(dest_subfolder_path, exist_ok=True)

            # subfolder 내의 모든 이미지 파일을 정렬하여 탐색합니다.
            image_files = sorted(os.listdir(subfolder_path))
            output_index = 1

            for image_file in image_files: #3개
                input_image_path = os.path.join(subfolder_path, image_file)
                
                if os.path.isfile(input_image_path):
                    # 출력 이미지 경로를 설정합니다.
                    t_frame_output_path = os.path.join(dest_subfolder_path, f"{output_index:02d}.png")
                    t1_frame_output_path = os.path.join(dest_subfolder_path, f"{output_index + 1:02d}.png")
                    
                    # tff_two_time 함수를 호출하여 이미지를 분리하고 저장합니다.
                    tff_two_time(input_image_path, t_frame_output_path, t1_frame_output_path)
                    
                    output_index += 2


#수행
# ori_path = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/gt_val_winter_two_time"
# dest_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_tff"
# process_folders_tff(ori_path,dest_path)




def process_folders_bff(ori_path, des_path):
    """
    ori_path 안의 subfolder들을 탐색하여 각 subfolder 안의 이미지를 처리하고,
    des_path에 새로운 subfolder를 만들어서 결과 이미지를 저장합니다.

    :param ori_path: 원본 폴더 경로
    :param des_path: 결과를 저장할 폴더 경로
    """
    # 원본 폴더 내의 모든 subfolder를 탐색합니다.
    for subfolder_name in sorted(os.listdir(ori_path)):
        subfolder_path = os.path.join(ori_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            # 결과 폴더를 생성합니다.
            dest_subfolder_path = os.path.join(des_path, subfolder_name)
            os.makedirs(dest_subfolder_path, exist_ok=True)

            # subfolder 내의 모든 이미지 파일을 정렬하여 탐색합니다.
            image_files = sorted(os.listdir(subfolder_path))
            output_index = 1

            for image_file in image_files: #3개
                input_image_path = os.path.join(subfolder_path, image_file)
                
                if os.path.isfile(input_image_path):
                    # 출력 이미지 경로를 설정합니다.
                    t_frame_output_path = os.path.join(dest_subfolder_path, f"{output_index:08d}.png")
                    t1_frame_output_path = os.path.join(dest_subfolder_path, f"{output_index+1:08d}.png")
                    
                    # tff_two_time 함수를 호출하여 이미지를 분리하고 저장합니다.
                    tff_two_time(input_image_path, t_frame_output_path, t1_frame_output_path)
                    
                    output_index += 2


#수행
# ori_path = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/gt_val_winter_two_time_fi_pad"
# dest_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_pad"
# process_folders_bff(ori_path,dest_path)


######################## 반으로 잘린 이미지 -> interpolate 해서 gt 스러운 이미지 만들기 ########################

def bicubic_interpolate_image(input_image_path, output_image_path, scale_factor=2):
    """
    주어진 이미지를 높이 방향으로 두 배로 bicubic 보간을 사용하여 확장하고 저장합니다.

    :param input_image_path: 원본 이미지 파일 경로
    :param output_image_path: 확장된 이미지 파일 경로
    :param scale_factor: 확장할 비율 (기본값: 2)
    """
    # 이미지를 읽어옵니다.
    image = Image.open(input_image_path)

    # 새로운 크기를 계산합니다.
    new_width = image.width
    new_height = image.height * scale_factor

    # 이미지를 bicubic 보간을 사용하여 크기 조정합니다.
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)

    # 확장된 이미지를 저장합니다.
    resized_image.save(output_image_path)
    
    

def process_folders_for_interpolation(ori_path, des_path, scale_factor=2):
    """
    ori_path 안의 subfolder들을 탐색하여 각 subfolder 안의 이미지를 높이 방향으로 두 배로
    bicubic 보간을 사용하여 확장하고, des_path에 저장합니다.

    :param ori_path: 원본 폴더 경로
    :param des_path: 결과를 저장할 폴더 경로
    :param scale_factor: 확장할 비율 (기본값: 2)
    """
    # 원본 폴더 내의 모든 subfolder를 탐색합니다.
    for subfolder_name in sorted(os.listdir(ori_path)):
        subfolder_path = os.path.join(ori_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            # 결과 폴더를 생성합니다.
            dest_subfolder_path = os.path.join(des_path, subfolder_name)
            os.makedirs(dest_subfolder_path, exist_ok=True)

            # subfolder 내의 모든 이미지 파일을 정렬하여 탐색합니다.
            image_files = sorted(os.listdir(subfolder_path))

            for image_file in image_files:
                input_image_path = os.path.join(subfolder_path, image_file)
                
                if os.path.isfile(input_image_path):
                    # 출력 이미지 경로를 설정합니다.
                    output_image_path = os.path.join(dest_subfolder_path, image_file)
                    
                    # bicubic_interpolate_image 함수를 호출하여 이미지를 확장하고 저장합니다.
                    bicubic_interpolate_image(input_image_path, output_image_path, scale_factor)

# 사용

# ori_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_pad'
# des_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_pad'
# process_folders_for_interpolation(ori_path, des_path)


# ori_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_pad'
# des_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/gt_val_winter_two_time_fi_bff_pad'
# process_folders_for_interpolation(ori_path, des_path)


######################## train data interpolate 한거에서, 최종적으로 even, odd, even, odd 해서 가져오기  ########################

"""

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
            # processed_img = img[1::2, :] if i % 2 == 0 else img[::2, :] #val_ version1 #try1 (이게 맞을 듯)
            # processed_img = img[::2, :] if i % 2 == 0 else img[1::2, :] #val_ version1 #try2 
            processed_img = img[1::2, :] if (i // 2) % 2 == 0 else img[::2, :] #try3 #eeooee,eeooee,eeooee,
            frames_output.append(processed_img)

        return frames_output
    
    
class Slice_pred2(nn.Module):

    def __init__(self):
        super(Slice_pred2, self).__init__()

    def forward(self, image, f_idx):
        frames_output = []
        # gt_frames_output=[]

        # Process frames for odd and even fields and save them
        for i, img in enumerate(image):
            # Process for odd and even fields
            # processed_img = img[1::2, :] if i % 2 == 0 else img[::2, :] #val_ version1 #try1 (이게 맞을 듯)
            # processed_img = img[::2, :] if i % 2 == 0 else img[1::2, :] #val_ version1 #try2 
            # processed_img = img[1::2, :] if (i // 2) % 2 == 0 else img[::2, :] #try3 #eeooee,eeooee,eeooee,
            #try4 #eeooee,ooeeoo,eeooee,
            if (i // 2) % 2 == 0 and f_idx % 2 == 0:
                processed_img= img[1::2, :]
            elif (i // 2) % 2 == 1 and f_idx % 2 == 0:
                processed_img= img[::2, :]
                
            elif (i // 2) % 2 == 0 and f_idx % 2 == 1:
                processed_img= img[::2, :]
            elif (i // 2) % 2 == 1 and f_idx % 2 == 1:
                processed_img= img[1::2, :]
            
            frames_output.append(processed_img)

        return frames_output
    


# Define source directory
# source_dir = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic"
source_dir = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_pad"

# Define destination directory
#destination_dir = os.path.join(os.getcwd(), 'train_sliced')
#gt_destination_dir = os.path.join(os.getcwd(), 'gt_sliced')

# destination_dir1="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_fi" #try1 (이게 맞을 듯)
# destination_dir1="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_fi2" #try2
# destination_dir1="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_fi3" #try3
# destination_dir1="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_fi4" #try4
destination_dir1="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_fi_bff_bicubic_pad_fi" #try5
# gt_destination_dir="C:\\Users\\인쇼츠\\Desktop\\Deinterlacing구현\\inference_data\\train\\img\\gt_sliced\\21_21"

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir1):
    os.makedirs(destination_dir1)


    
# Iterate over each folder in the source directory
for folder_idx, folder_name in sorted(enumerate(os.listdir(source_dir))):  #only first 72436 folder
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
        # slice_model = Slice_pred()
        slice_model2 = Slice_pred2()
               
        
        
        
        # Process images using Slice model
        # processed_images, gt_processed_images = slice_model(images)
        # processed_images = slice_model(images)
        processed_images = slice_model2(images, folder_idx)

        # Save processed images to the destination folder
        for i, processed_img in enumerate(processed_images):
            processed_img = Image.fromarray(processed_img)
            processed_img.save(os.path.join(destination_folder_path1, f'im{i+1}.png'),compress_level=0)
            

"""
######################## 만들어진 60개에서 30개씩만 가져와서 영상 말기 (1,3,5,만 가져오기    &     2,4,6만 가져오기 )  ########################

# List and sort files in the input directory
input_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi'
files = sorted(os.listdir(input_folder_path))

#### try 1

# output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi_only30_1'
# os.makedirs(output_folder_path, exist_ok=True)

# # Enumerate through the files and copy every second one
# i=1
# for file_idx, filename in enumerate(files):
#     ori_file_path = os.path.join(input_folder_path, filename)
#     if file_idx % 2 == 0: 
#         dest_file_path = os.path.join(output_folder_path, f"{i:08d}.png")
#         shutil.copyfile(ori_file_path, dest_file_path)
#         i+=1

# #### try 2

# output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi_only30_2'
# os.makedirs(output_folder_path, exist_ok=True)

# i=1
# for file_idx, filename in enumerate(files):
#     ori_file_path = os.path.join(input_folder_path, filename)
#     if file_idx % 2 == 1: 
#         dest_file_path = os.path.join(output_folder_path, f"{i:08d}.png")
#         shutil.copyfile(ori_file_path, dest_file_path)
#         i+=1
        
######################## remove padding ########################
# input_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi_only30_1'
# output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi_only30_1_rmv_pad'
# os.makedirs(output_folder_path, exist_ok=True)


# frames = []
# for file_idx, filename in sorted(enumerate(os.listdir(input_folder_path))):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         file_path = os.path.join(input_folder_path, filename)
#         img = Image.open(file_path)
#         frames.append((filename, np.array(img)))  # Store filename along with the frame

# for filename, frame in frames:
#     print("filename",filename)
#     # Process for odd and even fields
#     processed_frame = frame[13:512-13, :, :]  # Adjust the slice as needed

#     # Convert the processed frame back to a PIL Image
#     processed_img = Image.fromarray(processed_frame)

#     # Construct the output filename
#     output_filename = filename  # Use original filename
#     output_path = os.path.join(output_folder_path, output_filename)

#     processed_img.save(output_path, compress_level=0)
        
        
input_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi_only30_2'
output_folder_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/train_val_winter_two_time_pad_nobicubic_rst_fi_only30_2_rmv_pad'
os.makedirs(output_folder_path, exist_ok=True)

frames=[]
for filename in sorted(os.listdir(input_folder_path)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(input_folder_path, filename)
        img = Image.open(file_path)
        frames.append((filename, np.array(img)))  # Store filename along with the frame
          
for filename, frame in frames:
    # Process for odd and even fields
    processed_frame = frame[13:512-13,:,:] #13,14,...498
    
    # Convert the processed frame back to a PIL Image
    processed_img = Image.fromarray(processed_frame)

    # Construct the output filename including both folder name and original filename
    output_filename = f"{filename}"  # Concatenate folder name and original filename
    output_path = os.path.join(output_folder_path, output_filename)

    processed_img.save(output_path, compress_level=0)
