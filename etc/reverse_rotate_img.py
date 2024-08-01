import os
from PIL import Image
import numpy as np
import cv2

def extract_number(directory):
    return int(directory)

def extract_number2(directory):
    parts=os.path.splitext(directory)[0]
    return int(parts)


################################################### reverse rotate ###################################################
def reverse_rotate(input_path,output_path, rotated):
    img_idx=1
    for subfolder in sorted(os.listdir(input_path), key=extract_number):
        subfolder_path = os.path.join(input_path, subfolder)
        for img in sorted(os.listdir(subfolder_path), key=extract_number2):
            img_path= os.path.join(subfolder_path, img)
            img_name = f"{img_idx:08d}.png"
            save_img_path = os.path.join(output_path, img_name)
            # Read the image
            with Image.open(img_path) as image:
                # Rotate the image
                reverse_degreee=360-rotated
                rotated_image = image.rotate(reverse_degreee, expand=True)
                print(save_img_path)
                rotated_image.save(save_img_path)
       
            img_idx +=1

#process

# input_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst"
# output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev0"  
# os.makedirs(output_path, exist_ok=True)   
# reverse_rotate(input_path, output_path, rotated=0)

# input_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_90_slice_rst"
# output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev90"     
# os.makedirs(output_path, exist_ok=True)   
# reverse_rotate(input_path, output_path, rotated=90)

# input_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_180_slice_rst"
# output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev180"     
# os.makedirs(output_path, exist_ok=True)
# reverse_rotate(input_path, output_path, rotated=180)

# input_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_270_slice_rst"
# output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev270"  
# os.makedirs(output_path, exist_ok=True)   
# reverse_rotate(input_path, output_path, rotated=270)

################################################### 60 fps -> 30 fps  ###################################################

import os
import shutil

def copy_every_second_image(input_folder_path, output_folder_path):
    # List and sort files in the input directory
    files = sorted(os.listdir(input_folder_path))
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Enumerate through the files and copy every second one
    i = 1
    for file_idx, filename in enumerate(files):
        ori_file_path = os.path.join(input_folder_path, filename)
        if file_idx % 2 == 0:
            dest_file_path = os.path.join(output_folder_path, f"{i:08d}.png")
            shutil.copyfile(ori_file_path, dest_file_path)
            i += 1

# Example usage
# input_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev0'
# output_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev0_only30'
# copy_every_second_image(input_folder_path, output_folder_path)
        
        
# input_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev90'
# output_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev90_only30'
# copy_every_second_image(input_folder_path, output_folder_path)

# input_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev180'
# output_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev180_only30'
# copy_every_second_image(input_folder_path, output_folder_path)

# input_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev270'
# output_folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev270_only30'
# copy_every_second_image(input_folder_path, output_folder_path)

################################################### 네개 폴더에 있는 것들 평균 짓기 ###################################################
def average_images(input_paths, output_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get list of all files in the first input directory
    files = sorted(os.listdir(input_paths[0]))

    for file_name in files: #get file name 
        images = []
        
        for input_path in input_paths:
            image_path = os.path.join(input_path, file_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                images.append(image)
            else:
                print(f"Image {image_path} not found.")
                break

        if len(images) == len(input_paths):
            # Calculate the average image
            average_image = np.mean(images, axis=0).astype(np.uint8)
            
            # Save the average image to the output path
            output_image_path = os.path.join(output_path, file_name)
            cv2.imwrite(output_image_path, average_image)
            print(f"Saved averaged image: {output_image_path}")
        else:
            print(f"Skipping {file_name} due to missing images in some input folders.")
            
#process
input_paths = [
    "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev0_only30",
    "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev90_only30",
    "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev180_only30",
    "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice_rst_rev270_only30"
]
output_path = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/rst_average_images"
average_images(input_paths, output_path)
