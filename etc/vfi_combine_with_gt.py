#원래 sliced input : 짝홀짝홀짝홀/짝홀짝홀짝홀/짝홀짝홀짝홀/짝홀짝홀짝홀/짝홀짝홀짝홀...
# vfi 결과는 전제 동영상 프레임 중에 첫번째 프레임은 날리고 두번째 프레임부터 수행함 
# 홀짝홀짝홀/짝홀짝홀짝홀/짝홀짝홀짝홀/... -> VFI -> 짝홀짝홀짝/홀짝홀짝홀짝/홀짝홀짝홀짝/...

#따라서 기존 이미지 (input) 에서 가져와야하는 것 : 홀짝홀짝홀/짝홀짝홀짝홀/짝홀짝홀짝홀/

#근데, 6개씩 하나의 폴더에 있는 것이 아니고, 4000장 이미지가 하나의 폴더에 다같이 있으니, 이미지 파일 index로 판단하기
"""
vfi = 4000장 이미지 모음 #/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_output_20240718_105546_fi_toonefolder
input이미지 = 4000장 이미지 모음 #/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_toonefolder_wo_first

if img_idx % 2 ==0 : #img_idx가 0,2,4,...번째 이미지:
    vfi[img_idx]에서는 짝수번째 "행"  가져오기
    input이미지[img_idx]에서는 홀수번째 "행"  가져오기
else: #img_idx가 1,3,5,...번째 이미지:
    vfi[img_idx]에서는 홀수번째 "행"  가져오기
    input이미지[img_idx]에서는 짝수번째 "행"  가져오기
    
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image

def extract_number3(directory):
    parts=directory.split('/')[-1]
    parts = os.path.splitext(parts)[0]
    return int(parts)


# 이미지 경로
vfi_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_output_20240718_105546_fi_toonefolder'
input_image_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_toonefolder_wo_first'
dest_path='/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_output_20240718_105546_fi_toonefolder_withgt'
# 이미지 리스트
vfi_images = sorted([os.path.join(vfi_path, img) for img in os.listdir(vfi_path) if img.endswith(('.png', '.jpg', '.jpeg'))], key=extract_number3)
input_images = sorted([os.path.join(input_image_path, img) for img in os.listdir(input_image_path) if img.endswith(('.png', '.jpg', '.jpeg'))], key=extract_number3)

# 이미지 개수 확인
print('len1',len(vfi_images)) #286
print('len2',len(input_images)) #4381

# 첫 번째 이미지의 크기 확인
sample_img = cv2.imread(vfi_images[0])
channels, height, width = sample_img.shape[2], sample_img.shape[0], sample_img.shape[1]
print("sample_img.shape",sample_img.shape) #sample_img.shape (240, 720, 3)

# 모든 이미지를 담을 텐서 초기화 (4000, channels, height, width)
vfi_tensor = torch.zeros(len(vfi_images), channels, height, width)
input_tensor = torch.zeros(len(input_images), channels, height, width)

# 이미지를 텐서에 채우기
for i in range(len(vfi_images)):
    vfi_img = torch.from_numpy(cv2.imread(vfi_images[i]))
    input_img = torch.from_numpy(cv2.imread(input_images[i]))
    
    vfi_tensor[i] = torch.permute(vfi_img, (2, 0, 1))  # (240, 720, 3) -> (3, 240, 720)
    input_tensor[i] = torch.permute(input_img, (2, 0, 1))  # (240, 720, 3) -> (3, 240, 720)

print("이미지 텐서화 완료")


for img_idx in range(len(vfi_images)):
    
    img_tensor = torch.zeros(channels, height*2, width)
    
    if img_idx % 2 == 0:  # img_idx가 0, 2, 4, ...번째 이미지
        # vfi[img_idx]에서는 짝수번째 "행" 가져오기
        # input이미지[img_idx]에서는 홀수번째 "행" 가져오기
        img_tensor[:, ::2] = vfi_tensor[img_idx]  # prediction으로 만든 픽셀들
        img_tensor[:, 1::2] = input_tensor[img_idx]  # 첫input
    else:  # img_idx가 1, 3, 5, ...번째 이미지
        # vfi[img_idx]에서는 홀수번째 "행" 가져오기
        # input이미지[img_idx]에서는 짝수번째 "행" 가져오기
        img_tensor[:, 1::2] = vfi_tensor[img_idx]  # prediction으로 만든 픽셀들
        img_tensor[:, ::2] = input_tensor[img_idx]  # 첫input
    pred = img_tensor

    pred = torch.clamp(pred,0,255)
    pred_i = np.array(pred.cpu().detach()).astype(np.uint8).transpose(1,2,0)[:,:,[2,1,0]]

    pred_i = Image.fromarray(pred_i)
    pred_file_path=os.path.join(dest_path, f'{img_idx}.png')
    pred_i.save(pred_file_path, compress_level=0)
