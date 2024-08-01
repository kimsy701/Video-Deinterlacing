import os
from PIL import Image

def extract_number(directory):
    return int(directory)

def extract_number2(directory):
    parts=os.path.splitext(directory)[0]
    return int(parts)

def rotate(input_path,output_path, rotate=90):
    for subfolder in sorted(os.listdir(input_path), key=extract_number):
        subfolder_path = os.path.join(input_path, subfolder)
        output_subfolder = os.path.join(output_path, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        for img in sorted(os.listdir(subfolder_path), key=extract_number2):
            img_path= os.path.join(subfolder_path, img)
            save_img_path = os.path.join(output_subfolder, img)
            # Read the image
            with Image.open(img_path) as image:
                # Rotate the image
                rotated_image = image.rotate(rotate, expand=True)
                rotated_image.save(save_img_path)
       
######################## rotate   ######################## 

# input_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic"


# # output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_90"     
# # rotate(input_path, output_path, rotate=90)

# output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_180"     
# rotate(input_path, output_path, rotate=180)

# output_path="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_270"     
# rotate(input_path, output_path, rotate=270)



######################## rotate 한거에서, 최종적으로 even, odd, even, odd 해서 가져오기  ########################


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
            processed_img = img[1::2, :] if i % 2 == 0 else img[::2, :] #val_ version1 #try1 (이게 맞을 듯)
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
    

def slice(source_dir, destination_dir1):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir1):
        os.makedirs(destination_dir1)


    slice_model = Slice_pred()
    # slice_model2 = Slice_pred2()
            
    # Iterate over each folder in the source directory
    for folder_idx, folder_name in sorted(enumerate(os.listdir(source_dir))):  
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


            # Process images using Slice model
            processed_images = slice_model(images)
            # processed_images = slice_model2(images, folder_idx)

            # Save processed images to the destination folder
            for i, processed_img in enumerate(processed_images):
                processed_img = Image.fromarray(processed_img)
                processed_img.save(os.path.join(destination_folder_path1, f'im{i+1}.png'),compress_level=0)
            
            
###process###
# Define source directory
source_dir1 = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic"
destination_dir1="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_slice" 
slice(source_dir1, destination_dir1)

source_dir2 = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_90"
destination_dir2="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_90_slice" 
slice(source_dir2, destination_dir2)

source_dir3 = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_180"
destination_dir3="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_180_slice" 
slice(source_dir3, destination_dir3)

source_dir4 = "/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_270"
destination_dir4="/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/rotate_90_test/train_val_winter_two_time_fi_bff_bicubic_270_slice" 
slice(source_dir4, destination_dir4)
