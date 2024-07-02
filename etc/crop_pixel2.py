import cv2
import glob
import os
from tqdm import tqdm

def crop_pixel2(org_dir, save_dir, cut_index, ori_shape):  #약 1.6s/이미지 
    
    up_pixels, down_pixels, left_pixels, right_pixels =cut_index# 필요없는 부분: 위에서 부터 279 픽셀, 아래에서 부터 277 픽셀, (279, 277,0,0) 
    ori_w, ori_h = ori_shape #(3840, 2160)
        
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
        
    for folder_name in tqdm(os.listdir(org_dir)):
        folder_path = os.path.join(org_dir, folder_name)
        if os.path.isdir(folder_path):
            for i, img_path in enumerate(sorted(glob.glob(os.path.join(folder_path, '*.png')))):

                #img = cv2.imread(img_path) # Open image
                img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) #이렇게 해야 32비트로 읽어와서 32비트로 저장됨

                
                cropped_img = img[up_pixels:ori_h-down_pixels,left_pixels:ori_w-right_pixels]
                filename = os.path.splitext(os.path.basename(img_path))[0]

                save_path = os.path.join(save_dir, folder_name, f'{filename}.png') # Save cropped image
                os.makedirs(os.path.join(save_dir, folder_name), exist_ok=True)
                cv2.imwrite(save_path,cropped_img )

            print(f"Cropped image saved at {folder_path}")
        
#crop_pixel2('/mnt/sde/2D_Ani/스즈메의문단속/PNG_dataset','/mnt/sde/2D_Ani/스즈메의문단속/PNG_dataset_crop2',(279, 277,0,0),(3840, 2160) )
#crop_pixel2('/mnt/sda/2D_Ani/너의이름은/PNG_dataset','/mnt/sdd/2D_Ani/너의이름은/PNG_dataset_crop2',(1, 1,1,1),(3840, 2160) )
# crop_pixel2('/mnt/sda/2D_Ani/너의이름은/PNG_dataset','/mnt/sdd/2D_Ani/test/dataset',(1,1,1,1),(3840, 2160) )
crop_pixel2('/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/4K_바비/4k_바비_ori','/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/4k_바비_cropped',(840,840,1600,1600),(3840, 2160) )
