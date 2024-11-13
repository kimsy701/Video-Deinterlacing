import os
import cv2
import numpy as np
from tqdm import tqdm

#mean 취할 행들 txt에서 가져오기
def rows_from_txt(txt_path):
    # Read txt file and put row numbers in a list
    with open(txt_path, 'r') as file:
        rows = file.readlines()
    # Convert each line into an integer and strip newline characters
    li = [int(row.strip()) for row in rows]
    return li

def row_median(img, row_number):
    #print("img.shape",img.shape) #(486, 720, 3)
    # Ensure the row_number is within the valid range of rows in the image
    if 1 <= row_number < img.shape[0] - 1:
        img[row_number] = np.mean([img[row_number - 1], img[row_number + 1]], axis=0)
    return img

def row_median_include_itself(img, row_number):
    #print("img.shape",img.shape) #(486, 720, 3)
    # Ensure the row_number is within the valid range of rows in the image
    if 1 <= row_number < img.shape[0] - 1:
        img[row_number] = np.mean([img[row_number - 1], img[row_number] ,img[row_number + 1]], axis=0)
    return img
    
def row_median_weighted(img, row_number, weights=[0.1, 0.8, 0.1]): # weights=[0.25, 0.5, 0.25]):
    # Ensure the row_number is within the valid range of rows in the image
    if 1 <= row_number < img.shape[0] - 1:
        img[row_number] = np.average([img[row_number - 1], img[row_number], img[row_number + 1]], axis=0, weights=weights)
    return img

def row_median_weighted2(img, row_number, column_number, mid_weight=0.8): 
    # print(img.shape) #(486, 720, 3)
    
    diff_up = np.mean([
        np.abs(img[row_number-1, column_number-1] - img[row_number, column_number-1]),
        np.abs(img[row_number-1, column_number] - img[row_number, column_number]),
        np.abs(img[row_number-1, column_number+1] - img[row_number, column_number+1])
    ])
    
    
    diff_down = np.mean([
            np.abs(img[row_number+1, column_number-1] - img[row_number, column_number-1]),
            np.abs(img[row_number+1, column_number] - img[row_number, column_number]),
            np.abs(img[row_number+1, column_number+1] - img[row_number, column_number+1])
        ])    
    
    # Ensure the row_number is within the valid range of rows in the image
    if  diff_up > diff_down:
        weights=[0.07,0.8,0.13]
    else: 
        weights=[0.13,0.8,0.07]
    
    if 1 <= row_number < img.shape[0] - 1:
        img[row_number, column_number] = np.average([img[row_number - 1, column_number], img[row_number,column_number], img[row_number + 1,column_number]], axis=0, weights=weights)
    return img
    
    
# ######## process #######
folder_path = '/mnt/sda/vsr_dataset/MZDV_8001_Hoshinokoe_20121201_main'
dest_path = '/mnt/sda/vsr_dataset/MZDV_8001_Hoshinokoe_20121201_main_0.1_0.8_0.1_diffweight_16bit'
txt_path = '/mnt/sda/vsr_dataset/row_median.txt'


# #Create the destination directory if it doesn't exist
# os.makedirs(dest_path, exist_ok=True)


# for img_name  in tqdm(sorted(os.listdir(folder_path))):
#     img_path = os.path.join(folder_path, img_name )
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
#     h,w = img.shape[0], img.shape[1]
    
#     li = rows_from_txt(txt_path)
    
#     for row_num in li:
#         # img = row_median(img, row_num)
#         # img = row_median_include_itself(img, row_num)
#         img = row_median_weighted(img, row_num, weights=[0.1,0.8,0.1])
        
#     #위아래 차이 따져서 weight 주기
#         for col_num in range(1, w-1):
#             img = row_median_weighted2(img, row_num, col_num, mid_weight=0.8) 
    
#     fi_img_path = os.path.join(dest_path, img_name)
#     cv2.imwrite(fi_img_path, img)
    
    
    
    
######## process 전체  #######

folder_path = '/mnt/sda/vsr_dataset/MZDV_8001_Hoshinokoe_20121201_main'
dest_path = '/mnt/sda/vsr_dataset/MZDV_8001_Hoshinokoe_20121201_main_0.1_0.8_0.1_diffweight_16bit'
txt_path = '/mnt/sda/vsr_dataset/row_median.txt'


# Create the destination directory if it doesn't exist
os.makedirs(dest_path, exist_ok=True)

    
for img_name in tqdm(sorted(os.listdir(folder_path))):
    img_path = os.path.join(folder_path, img_name )
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    h,w = img.shape[0], img.shape[1]
    
    li = rows_from_txt(txt_path)
    
    for row_num in li:
        img = row_median_weighted(img, row_num,weights=[0.1, 0.8, 0.1])
        
    #위아래 차이 따져서 weight 주기
    for row_num in li:
        for col_num in range(1, w-1):
            img = row_median_weighted2(img, row_num, col_num, mid_weight=0.8) 
    
    fi_img_path = os.path.join(dest_path, img_name)
    cv2.imwrite(fi_img_path, img,[cv2.IMWRITE_TIFF_COMPRESSION, 1])
    
