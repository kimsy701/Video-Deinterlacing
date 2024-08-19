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
    
def row_median_weighted(img, row_number, weights=[0.25, 0.5, 0.25]): # weights=[0.25, 0.5, 0.25]):
    # Ensure the row_number is within the valid range of rows in the image
    if 1 <= row_number < img.shape[0] - 1:
        img[row_number] = np.average([img[row_number - 1], img[row_number], img[row_number + 1]], axis=0, weights=weights)
    return img

    
    
######## process #######
folder_path = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_1min_re'
#dest_path = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_1min_mean_re'
#dest_path = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_1min_mean_include_itself_re'
#dest_path = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_1min_mean_weighted_0.25_0.5_0.25_re'
dest_path = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_1min_mean_weighted_0.3_0.4_0.3_re'
txt_path = '/mnt/sdb/VSR_Inference/row_median.txt'


# Create the destination directory if it doesn't exist
# os.makedirs(dest_path, exist_ok=True)


# for img_name  in tqdm(os.listdir(folder_path)):
#     img_path = os.path.join(folder_path, img_name )
#     img = cv2.imread(img_path)
    
#     li = rows_from_txt(txt_path)
    
#     for row_num in li:
#         #img = row_median(img, row_num)
#         #img = row_median_include_itself(img, row_num)
#         img = row_median_weighted(img, row_num)
    
#     fi_img_path = os.path.join(dest_path, img_name)
#     cv2.imwrite(fi_img_path, img)
    
    
    
    
######## process 전체  #######

folder_path = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main'
dest_path1 = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_mean_include_itself'
dest_path2 = '/mnt/sdb/VSR_Inference/MZDV_8001_Hoshinokoe_20121201_main_mean_weighted_0.25_0.5_0.25'
txt_path = '/mnt/sdb/VSR_Inference/row_median.txt'


# Create the destination directory if it doesn't exist
os.makedirs(dest_path1, exist_ok=True)

for img_name  in tqdm(os.listdir(folder_path)):
    img_path = os.path.join(folder_path, img_name )
    img = cv2.imread(img_path)
    
    li = rows_from_txt(txt_path)
    
    for row_num in li:
        #img = row_median(img, row_num)
        img = row_median_include_itself(img, row_num)
        #img = row_median_weighted(img, row_num)
    
    fi_img_path = os.path.join(dest_path1, img_name)
    cv2.imwrite(fi_img_path, img)
    
    
os.makedirs(dest_path2, exist_ok=True)

for img_name  in tqdm(os.listdir(folder_path)):
    img_path = os.path.join(folder_path, img_name )
    img = cv2.imread(img_path)
    
    li = rows_from_txt(txt_path)
    
    for row_num in li:
        #img = row_median(img, row_num)
        #img = row_median_include_itself(img, row_num)
        img = row_median_weighted(img, row_num) #0.25,0.5,0.25
    
    fi_img_path = os.path.join(dest_path2, img_name)
    cv2.imwrite(fi_img_path, img)
    
    
