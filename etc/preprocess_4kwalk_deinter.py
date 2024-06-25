import os
import shutil
from PIL import Image
import numpy as np
import cv2
import argparse
import torch.nn as nn
import re
import subprocess



def natural_sort_key(entry):
    # Split the filename into numeric and non-numeric parts
    name_parts = entry.name.split('.') 
    # if len(name_parts) < 2:
    #     return entry.name
    filename, ext = '.'.join(name_parts[:-1]), name_parts[-1]
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', filename)]

################################################### 1개의 이미지 4k에서 (640,480) 사이즈 4개 crop ###################################################
#A좌표 : (640,480,640x2,480x2)
#B좌표 : (640x2,480x2, 640x3,480x3)
#C좌표 : (640x3,480, 640x4,480x2)
#D좌표 : (640x4,480x2, 640x5,480x3)

def crop_shifted_images(image): 
    A_cor = (640,480,640*2,480*2)
    B_cor = (640*2,480*2, 640*3,480*3)
    C_cor = (640*3,480, 640*4,480*2)
    D_cor = (640*4,480*2, 640*5,480*3)
    
    """Generate shifted images towards the center."""
    A_img=image[A_cor[1]:A_cor[3], A_cor[0]:A_cor[2], :] 
    B_img=image[B_cor[1]:B_cor[3], B_cor[0]:B_cor[2], :]
    C_img=image[C_cor[1]:C_cor[3], C_cor[0]:C_cor[2], :]
    D_img=image[D_cor[1]:D_cor[3], D_cor[0]:D_cor[2], :]
    

    return A_img, B_img, C_img,D_img


################################################### 전체 png들을 6개씩 나눠서 각각의 폴더(1,2,...)에 담기. 그 후 A,B,C,D로 crop해서 각각 4개의 폴더에 저장  ###################################################
def organize_and_crop_images(src_path1, dest_path1, batch_size=6):
    batch = []
    batch_index = 1
    current_batch_count = 0

    with os.scandir(src_path1) as entries:
        print("type", type(entries))
        sorted_entries = sorted(entries, key=natural_sort_key)
        for entry in sorted_entries:
            batch.append(entry.name)
            print("batch", batch)
            current_batch_count += 1

            if current_batch_count == batch_size:
                process_batch(batch, src_path1, dest_path1, batch_index)
                batch_index += 1
                batch = []
                current_batch_count = 0

    # Process any remaining files in the last batch
    if batch:
        process_batch(batch, src_path1, dest_path1, batch_index)

def process_batch(batch, src_folder, dest_path, batch_index):
    
    # for file_name in sorted(batch, key=natural_sort_key):
    for file_name in batch:
        src_file_path = os.path.join(src_folder, file_name)
        png_img = cv2.imread(src_file_path, cv2.IMREAD_UNCHANGED)
        A_png_img, B_png_img, C_png_img, D_png_img = crop_shifted_images(png_img)
        
        for label, img in zip("ABCD", [A_png_img, B_png_img, C_png_img, D_png_img]):
            output_dir = os.path.join(dest_path,f'{batch_index}_{label}')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, img)
            print(f'Cropped {file_name} into {label} and saved to {output_path}')


################################################### 8의 배수의 크기로 resize ###################################################
#486 -> 위 아래 3,3 씩 자르기 
#/home/kimsy701/deinter_venv/train_val_MS 에서 /home/kimsy701/deinter_venv/train_val_MS_crop으로 
input_folder_path='/home/kimsy701/deinter_venv/train_val_winter1080'
output_folder_path='/home/kimsy701/deinter_venv/train_val_winter_crop1080'

for folder in sorted(os.listdir(input_folder_path)):
    frames = [] #for each 1_1, 1_2 folder
    folder_path = os.path.join(input_folder_path, folder)
    
    for filename in sorted(os.listdir(folder_path)):
      if filename.endswith(('.png', '.jpg', '.jpeg')):
          file_path = os.path.join(folder_path, filename)
          img = Image.open(file_path)
          frames.append((filename, np.array(img)))  # Store filename along with the frame
          
    for filename, frame in frames:
      # Process for odd and even fields
        processed_frame = frame[3:483,:,:] #3,4,5....482
        
        # Convert the processed frame back to a PIL Image
        processed_img = Image.fromarray(processed_frame)

        # Construct the output filename including both folder name and original filename
        output_filename = f"{filename}"  # Concatenate folder name and original filename
        output_path = os.path.join(output_folder_path, folder, output_filename)

        if not os.path.exists(os.path.join(output_folder_path, folder)):
            os.makedirs(os.path.join(output_folder_path, folder))

        processed_img.save(output_path)


################################################### 6장 내에서 어느 2개의 min(flow) > mean(vimeo) 인 6개 set 데이터셋만 가져오기  ###################################################

################################################### (slice): 6개 단위로 1,3,5는 even만, 2,4,6은 odd만 가져오기 ###################################################

class Slice_pred(nn.Module):

    def __init__(self):
        super(Slice_pred, self).__init__()

    def forward(self, image):
        frames_output = []
        # gt_frames_output=[]

        # Process frames for odd and even fields and save them
        for i, img in enumerate(image):
            # Process for odd and even fields
            processed_img = img[1::2, :] if i % 2 == 0 else img[::2, :] #val_ version1
            frames_output.append(processed_img)

        return frames_output
    

def slice_image(src_path2,dest_path2): 

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_path2):
        os.makedirs(dest_path2)


    # Iterate over each folder in the source directory
    for folder_name in sorted(os.listdir(src_path2)):  
        if os.path.isdir(os.path.join(src_path2, folder_name)):
            # Construct source folder path
            source_folder_path = os.path.join(src_path2, folder_name)
            # Construct destination folder path
            destination_folder_path2 = os.path.join(dest_path2, folder_name)
            # Create destination folder if it doesn't exist
            if not os.path.exists(destination_folder_path2):
                os.makedirs(destination_folder_path2)


            # Load images from the source folder
            images = []  # List to hold images as arrays for processing
            for filename in sorted(os.listdir(source_folder_path)):
                if filename.endswith('.png'):  # Assuming images are PNG files
                    img = Image.open(os.path.join(source_folder_path, filename))
                    images.append(np.array(img))

            # Instantiate Slice model
            slice_model = Slice_pred()
            # Process images using Slice model
            # processed_images, gt_processed_images = slice_model(images)
            processed_images = slice_model(images)

            # Save processed images to the destination folder
            for i, processed_img in enumerate(processed_images):
                processed_img = Image.fromarray(processed_img)
                processed_img.save(os.path.join(destination_folder_path2, f'im{i+1}.png'))
                


    print("Images processed and saved successfully!")

#저장


################################################### main###################################################
def main():
    parser = argparse.ArgumentParser(description='Process source and destination paths.')
    parser.add_argument('src_path1', type=str, help='Path of the source image for function 1')
    parser.add_argument('dest_path1', type=str, help='Path of the destination for function 1')
    parser.add_argument('dest_path2', type=str, help='Path of the destination for function 2')
    
    args = parser.parse_args()

    organize_and_crop_images(args.src_path1, args.dest_path1)
    slice_image(args.dest_path1, args.dest_path2)

if __name__ == '__main__':
    main()
    
    
#CMD code : python image_processing.py /path/to/src_image1 /path/to/dest_folder1 /path/to/dest_folder2
# python preprocess_4kwalk_deinter.py /home/nick/inshorts_vsr/hy/test /home/nick/inshorts_vsr/hy/test_dest_path1 /home/nick/inshorts_vsr/hy/test_dest_path2
