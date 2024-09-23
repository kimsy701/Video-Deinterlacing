import os
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


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

def calculate_ssim(img1, img2):
    # img1과 img2는 numpy 배열 형태의 이미지
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SSIM 계산
    ssim_scr, _ = ssim(gray_img1, gray_img2, full=True)
    return ssim_scr

def calculate_sobel(image, threshold=100):
    # Check if image is loaded correctly
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Apply Sobel operator in the x direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Apply threshold to keep only significant gradients
    sobel_magnitude = np.where(sobel_magnitude > threshold, sobel_magnitude, 0)

    
    # Calculate the sum of significant gradient magnitudes
    edge_score = np.sum(sobel_magnitude)
    
    # Normalize by the number of pixels to prevent size bias
    normalized_edge_score = edge_score / image.size
    
    # return normalized_edge_score, sobel_magnitude
    return normalized_edge_score


def calculate_sobel_gpu(image, threshold=100):
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Convert image to grayscale and upload it to the GPU
    image_gpu = cv2.cuda_GpuMat()
    image_gpu.upload(image)
    
    # Apply Sobel operator on GPU
    sobel_x_gpu = cv2.cuda.Sobel(image_gpu, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_gpu = cv2.cuda.Sobel(image_gpu, cv2.CV_64F, 0, 1, ksize=3)
    
    # Download results back to CPU
    sobel_x = sobel_x_gpu.download()
    sobel_y = sobel_y_gpu.download()
    
    # Compute magnitude of gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Apply threshold
    sobel_magnitude = np.where(sobel_magnitude > threshold, sobel_magnitude, 0)

    # Calculate and normalize edge score
    edge_score = np.sum(sobel_magnitude)
    normalized_edge_score = edge_score / image.size
    
    return normalized_edge_score

def psnr_ssim_sobel_edge(predicted_path, get_path_names):

    file_length = len(os.listdir(gt_path))

    total_psnr=[]
    total_ssim=[]
    total_sobel=[]

    for file_index in tqdm(range(file_length)):
        pred_img_path = os.path.join(predicted_path, sorted(os.listdir(predicted_path))[file_index])   #폴더들 이름에 따라 sorting 잘 됐는지 확인하기 
        gt_img_path =  os.path.join(gt_path, sorted(os.listdir(gt_path))[file_index]) #폴더들 이름에 따라 sorting 잘 됐는지 확인하기 
        
        print("pred_img_path",pred_img_path)
        print("gt_img_path",gt_img_path)

        
        pred_img= Image.open(pred_img_path)
        gt_img= Image.open(gt_img_path)
        
        pred_img = np.array(pred_img)
        gt_img = np.array(gt_img)
        

        psnr = calculate_psnr(pred_img,gt_img)
        print("psnr",psnr)
        ssim = calculate_ssim(pred_img,gt_img)
        print("ssim", ssim)
        sobelscr = calculate_sobel(pred_img,gt_img)
        print("sobelscr",sobelscr)

        total_psnr.append(psnr)
        total_ssim.append(ssim)
        total_sobel.append(sobelscr)
        
    
    # with open("/home/kimsy701/training_code/training_code_fi/training_code_fi_fi/other_psnr_main_list_fi.txt", 'w') as f:
    with open("/mnt/sda/deinter_datasets/LIU4k_v2/scores/BASICVSRPP_psnr_ssim_sobel.txt", 'w') as f:
        for i in range(len(total_psnr)):
            f.write(f"{total_psnr[i]}, {total_ssim[i]}, {total_sobel[i]}\n")
      
##### 실행 #####        
predicted_path= "/mnt/sdb/VSR_Inference/LIU4k_v2_val_100_fi_4.5downsize_basicvsr_pp_inter4k_Raft_less_deg_x4_gloss_240807_large_patch_scratch_128_model_300"
gt_path="/mnt/sda/deinter_datasets/LIU4k_v2/val_100_fi_resize"


psnr_ssim_sobel_edge(predicted_path, gt_path)
        
