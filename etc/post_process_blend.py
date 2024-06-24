import os
import cv2
import numpy as np
import shutil

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

def apply_edge_smoothing(image, method='default', ksize=5, sigma=1.0):
    if method == 'default':
        smoothed = cv2.GaussianBlur(image, (ksize, ksize), sigma/4) #sigma에서 sigma/2해서 강도 줄임
    elif method == 'sharpness':
        smoothed = cv2.addWeighted(image, 1.2, cv2.GaussianBlur(image, (ksize, ksize), sigma/4), -0.2, 0) #파라미터들 조정하여 강도 줄임
    elif method == 'smode':
        smoothed = cv2.bilateralFilter(image, d=ksize, sigmaColor=15, sigmaSpace=15) #sigma를 75에서 30으로 줄여 강도를 줄임
    elif method == 'slmode':
        smoothed = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21) #잡음 제거 강도를 줄임
    elif method == 'tr2':
        smoothed = image.copy()
        for _ in range(2):
            smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), sigma/4) #sigma에서 sigma/2해서 강도 줄임
    else:
        raise ValueError("Invalid method specified. Choose from 'default', 'sharpness', 'smode', 'slmode', 'tr2'.")
    return smoothed

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


################################################################### Make blended images ###################################################################
def make_belnded_images(input_folder,output_folder, methods):
    for subfolders in sorted(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolders)
        
        output_subfolder = os.path.join(output_folder, subfolders)
        os.makedirs(output_subfolder, exist_ok=True)
        # psnr_results=[]
        
        for filename in sorted(os.listdir(subfolder_path)):
            input_image_path = os.path.join(input_folder, subfolders, filename)
            # ground_truth_image_path = os.path.join(ground_truth_folder, subfolders, filename)

            # Load the images
            image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
            # ground_truth = cv2.imread(ground_truth_image_path,cv2.IMREAD_UNCHANGED)

            # sub_psnr_results=[]

            for method in methods:
                method_path = os.path.join(output_folder,subfolders, method)
                os.makedirs(method_path,exist_ok=True)
                fi_method_path = os.path.join(method_path, filename)
                smoothed_image = apply_edge_smoothing(image, method=method)
                cv2.imwrite(fi_method_path, smoothed_image)
                # psnr = calculate_psnr(smoothed_image, ground_truth)
                # sub_psnr_results.append(psnr)
            
################################################################### calculate psnr ###################################################################
def exec_calculate_psnr(pred_path, gt_path):
    total_psnr=[]
    gt_subfolder_list=sorted(os.listdir(gt_path),key = extract_number)
    
    for i,subfolders in enumerate(sorted(os.listdir(pred_path))):
        method_psnr = []
        filenames=[]
        
        for method_folder in sorted(os.listdir(os.path.join(pred_path, subfolders))):
            
            six_psnr = []
            
            for f,filename in enumerate(sorted(os.listdir(os.path.join(pred_path, subfolders,method_folder)))):
                filenames.append(sorted(os.listdir(os.path.join(gt_path, gt_subfolder_list[i])))[f])

                pred = cv2.imread(os.path.join(pred_path, subfolders,method_folder,filename), cv2.IMREAD_UNCHANGED) #read predicted file by file name
                gt = cv2.imread(os.path.join(gt_path, gt_subfolder_list[i],filenames[f]), cv2.IMREAD_UNCHANGED) 
                
                
                psnr = calculate_psnr(pred, gt)
                six_psnr.append(psnr)
            method_psnr.append(sum(six_psnr)/len(six_psnr))
        total_psnr.append(method_psnr)
                
    if i==len(os.listdir(pred_path))-1:
        with open("/home/kimsy701/training_code/training_code_fi/training_code_fi_fi/QTGMC_blended_psnr_list.txt", 'w') as f:
            for m,method_psnr in enumerate(total_psnr):
                f.write(f"image set{m}"+str(method_psnr)+"\n")
                
def exec_calculate_psnr_wo_method(pred_path, gt_path):
    total_psnr=[]
    gt_subfolder_list=sorted(os.listdir(gt_path),key = extract_number)
    
    for i,subfolders in enumerate(sorted(os.listdir(pred_path))):
        method_psnr = []
        filenames=[]
        six_psnr=[]
        for f,filename in enumerate(sorted(os.listdir(os.path.join(pred_path, subfolders)))):
            filenames.append(sorted(os.listdir(os.path.join(gt_path, gt_subfolder_list[i])))[f])

            pred = cv2.imread(os.path.join(pred_path, subfolders,filename), cv2.IMREAD_UNCHANGED) #read predicted file by file name
            gt = cv2.imread(os.path.join(gt_path, gt_subfolder_list[i],filenames[f]), cv2.IMREAD_UNCHANGED) 
            
            
            psnr = calculate_psnr(pred, gt)
            six_psnr.append(psnr)
        total_psnr.append(sum(six_psnr)/len(six_psnr))
            
    if i==len(os.listdir(pred_path))-1:
        with open("/home/kimsy701/training_code/training_code_fi/training_code_fi_fi/QTGMC_psnr_list.txt", 'w') as f:
            for i in total_psnr:
                f.write(str(i)+"\n")
            
########## if output files are in one whole folder, make it in each folder with 6 images each. ########################

def copy_images_to_folders(src_folder, dest_folder):
    # 원본 폴더의 이미지 파일들을 가져오기
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # 파일 이름 순서대로 정렬

    num_folders = (len(image_files) + 5) // 6  # 이미지 파일 수에 따른 폴더 개수 계산

    for folder_idx in range(num_folders):
        folder_name = f"{folder_idx}"
        folder_path = os.path.join(dest_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for i in range(6):
            img_index = folder_idx * 6 + i
            if img_index < len(image_files):
                src_file_path = os.path.join(src_folder, image_files[img_index])
                dest_file_path = os.path.join(folder_path, image_files[img_index])
                shutil.copy(src_file_path, dest_file_path)
    
                
################################################################### apply functions ###################################################################

# Process each image in the input folder
# methods = ['default', 'sharpness', 'smode', 'slmode', 'tr2']

# Paths
# input_folder = "/home/kimsy701/deinter_dataset/interpolate_train_val_100_rst"
# output_folder = "/home/kimsy701/deinter_dataset/interpolate_blend_train_val_100_rst"
# #make image
# make_belnded_images(input_folder,output_folder, methods)

#한 폴더에 이미지들이 한번에 있다면, 6개씩 묶어줘서 PSNR계산하자
# input_path = "/home/kimsy701/QTGMC/output"
# out_path = "/home/kimsy701/QTGMC/output_fi"
# copy_images_to_folders(input_path,out_path)

#calculate psnr
pred_path="/home/kimsy701/QTGMC/output_fi"
gt_path="/home/kimsy701/deinter_dataset/gt_val_100_re"
# exec_calculate_psnr(pred_path,gt_path)
exec_calculate_psnr_wo_method(pred_path,gt_path)
