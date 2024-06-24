import os
import numpy as np
from PIL import Image
import cv2


def extract_number(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        first_part =parts[0] #parts[-1]
        second_part = parts[-1]
        try:
            return int(first_part)*1000 +int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)S
    return float('inf')  # Return infinity if no underscore is found (optional)

def extract_number2(directory):
    return int(directory) 

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr



def slice_pred_gt(predicted_path, gt_path, save_pred_path,save_gt_path):

    folder_length = len(os.listdir(gt_path))
    total_mse=[]    
    total_psnr=[]

    for folder_index in range(folder_length):
        predicted_folder_path = os.path.join(predicted_path, sorted(os.listdir(predicted_path),key=extract_number2)[folder_index])   #폴더들 이름에 따라 sorting 잘 됐는지 확인하기 
        gt_folder_path =  os.path.join(gt_path, sorted(os.listdir(gt_path),key=extract_number)[folder_index]) #폴더들 이름에 따라 sorting 잘 됐는지 확인하기 
        
        # print("predicted_folder_path",predicted_folder_path)
        # print("gt_folder_path",gt_folder_path)
        
        save_pred_folder = os.path.join(save_pred_path, f"{folder_index}")
        save_gt_folder = os.path.join(save_gt_path, f"{folder_index}")
        
        os.mkdir(save_pred_folder)
        os.mkdir(save_gt_folder)
        
        six_mse=[]
        six_psnrs=[]
        
        for img_idx in range(6):
            pred_img_path = os.path.join(predicted_folder_path, sorted(os.listdir(predicted_folder_path))[img_idx])
            gt_img_path = os.path.join(gt_folder_path, sorted(os.listdir(gt_folder_path))[img_idx])
            
            
            pred_img= Image.open(pred_img_path)
            gt_img= Image.open(gt_img_path)
            
            pred_img = np.array(pred_img)
            gt_img = np.array(gt_img)
            
            # print("pred_img shape ",pred_img.shape) #pred_img shape  (256, 448, 3)
            h,w,c = pred_img.shape 
            # print(pred_img.shape,1) #(256, 448, 3)
            half_pred_array = np.ones((h//2,w,c),dtype=float)
            
            if img_idx % 2 == 0:
                # 짝수 인덱스의 이미지
                
                #이게 맞음    
                half_pred_array =pred_img[::2, :, :]   
                half_gt_array = gt_img[::2, :, :] #위와 상응하는 gt만 가져오기
                
                
                
                
            else:
                # 홀수 인덱스의 이미지
                #이게 맞음  
                half_pred_array = pred_img[1::2, :, :]
                half_gt_array = gt_img[1::2, :, :]#gt_img[1::2, :, :]   #위와 상응하는 gt만 가져오기
                
                
                
                
                
            # Create mask for differences
            diff_mask = half_pred_array != half_gt_array
            # print(diff_mask.shape) #(128, 448, 3) #(h,w,c)
            # Highlight differences in red on a copy of the predicted image
            highlighted_pred = half_pred_array.copy()
            # print(highlighted_pred[diff_mask].shape) (0,)
            # print("sum of highlighted_pred[diff_mask]", sum(highlighted_pred[diff_mask])) #0
        
            
            
                
            half_pred_path = os.path.join(save_pred_folder, f"{img_idx}.png")
            half_gt_path = os.path.join(save_gt_folder, f"{img_idx}.png")
        
            cv2.imwrite(half_pred_path, half_pred_array.astype(np.uint8)[:,:,::-1])
            cv2.imwrite(half_gt_path, half_gt_array.astype(np.uint8)[:,:,::-1])
            
            mse = np.mean((half_pred_array - half_gt_array) ** 2)    
            psnr = calculate_psnr(half_pred_array,half_gt_array)
            six_mse.append(mse)
            six_psnrs.append(psnr)
        avg_psnr = sum(six_mse)/len(six_mse)
        avg_psnr = sum(six_psnrs)/len(six_psnrs)
        total_mse.append(avg_psnr)
        total_psnr.append(avg_psnr)
        
    
    # with open("/home/kimsy701/training_code/training_code_fi/training_code_fi_fi/other_psnr_main_list_fi.txt", 'w') as f:
    with open("/home/kimsy701/training_code/training_code_fi/training_code_fi_fi/other_psnr_main_list1.txt", 'w') as f:
        for i in total_psnr:
            f.write(str(i)+"\n")
    with open("/home/kimsy701/training_code/training_code_fi/training_code_fi_fi/other_mse_main_list1.txt", 'w') as f:
        for i in total_mse:
            f.write(str(i)+"\n")
        
        
        
        
##### 실행 #####        
predicted_path= "/home/kimsy701/deinter_dataset/interpolate_train_val_100_rst"
gt_path="/home/kimsy701/deinter_dataset/gt_val_100_re"

save_pred_path="/home/kimsy701/inference결과들/interpolate_half_train_val_100_rst"
save_gt_path="/home/kimsy701/inference결과들/interpolate_half_gt_val_100_rst"

# os.mkdir(save_pred_path)
# os.mkdir(save_gt_path)

slice_pred_gt(predicted_path, gt_path, save_pred_path,save_gt_path)
        
