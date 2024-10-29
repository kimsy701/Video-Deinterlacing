import cv2
import os
from tqdm import tqdm

def extract_number(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        last_part =os.path.splitext(parts[-1])[0] 
        # print("first_part",first_part )

        second_last_part = parts[-2].split("rst")[-1]
        # print("second_part",second_part )
        try:
            return int(second_last_part)*100000 +int(last_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)


def extract_number2(directory):
    return int(directory)

def extract_number3(directory):
    parts = os.path.splitext(directory)[0]
    return int(parts)

def alphabet_to_number(letter):
    return ord(letter.upper()) - 64

def extract_number4(directory):
    parts = directory.split('_')
    
    part0 = parts[0][-2:]
    part1 = parts[1] #01
    part2 = alphabet_to_number(parts[2]) #A -> 1, B-> 2,...
    part3 = parts[3] #0000
    part4 = os.path.splitext(parts[4])[0][-1] #im1, im3 의 1,3 부분
        
    return int(part0)*1000000 + int(part1)*10000 + int(part2)*1000 + int(part3)*100 + int(part4)

def extract_number5(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        first_part =parts[0]
        # print("first_part",first_part )

        second_part = os.path.splitext(parts[-1])[0]
        # print("second_part",second_part )
        try:
            return int(first_part)*100000 +int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)


################################ slice to half height  ################################

def slice_height(folder_path, output_folder):
    # 폴더 내 모든 이미지 파일 순회
    for img_idx, filename in enumerate(sorted(os.listdir(folder_path))):
        image_path = os.path.join(folder_path, filename)
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Failed to read {filename}. Skipping...")
            continue

        print(f"Processing {filename} with shape {image.shape}")

        processed_frame1 = image[::2, :, :]
        processed_frame2 = image[1::2, :, :]

        # BGR 이미지를 RGB로 변환 (Matplotlib에서 제대로 보이도록)
        processed_frame1_rgb = cv2.cvtColor(processed_frame1, cv2.COLOR_BGR2RGB)
        processed_frame2_rgb = cv2.cvtColor(processed_frame2, cv2.COLOR_BGR2RGB)

        frame1_path = os.path.join(output_folder, f'{img_idx}_1.png')
        frame2_path = os.path.join(output_folder, f'{img_idx}_2.png')
        cv2.imwrite(frame1_path, cv2.cvtColor(processed_frame1_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(frame2_path, cv2.cvtColor(processed_frame2_rgb, cv2.COLOR_RGB2BGR))

        # print(f"Saved processed frames as {frame1_path} and {frame2_path}")



################################ height x2 by bicubic  ################################

# from PIL import Image 
# import os 
# from tqdm import tqdm

# def extract_number(directory):
#     parts = directory.split('_')
#     if len(parts) > 1:
#         first_part =parts[0]
#         # print("first_part",first_part )

#         second_part = os.path.splitext(parts[-1])[0]
#         # print("second_part",second_part )
#         try:
#             return int(first_part)*100000 +int(second_part)
#         except ValueError:
#             return float('inf')  # Return infinity for non-numeric parts (optional)
#     return float('inf')  # Return infinity if no underscore is found (optional)

# def bicubic_interpolate_image(input_image_path, output_image_path, scale_factor=2):
#     """
#     주어진 이미지를 높이 방향으로 두 배로 bicubic 보간을 사용하여 확장하고 저장합니다.

#     :param input_image_path: 원본 이미지 파일 경로
#     :param output_image_path: 확장된 이미지 파일 경로
#     :param scale_factor: 확장할 비율 (기본값: 2)
#     """
#     # 이미지를 읽어옵니다.
#     image = Image.open(input_image_path)

#     # 새로운 크기를 계산합니다.
#     new_width = image.width
#     new_height = image.height * scale_factor

#     # 이미지를 bicubic 보간을 사용하여 크기 조정합니다.
#     resized_image = image.resize((new_width, new_height), Image.BICUBIC)

#     # 확장된 이미지를 저장합니다.
#     resized_image.save(output_image_path)
    
# input_folder_path = '/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice'
# output_folder_path = '/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice_bicubic'

# for img in tqdm(sorted(os.listdir(input_folder_path), key = extract_number)):
#     input_image_path = os.path.join(input_folder_path, img)
#     output_image_path = os.path.join(output_folder_path, img)
#     bicubic_interpolate_image(input_image_path, output_image_path)


def bicubic_interpolate_image(input_image_path, output_image_path, scale_factor=2):
    """
    Reads a 16-bit image, scales it vertically by a given factor using bicubic interpolation,
    and saves the output image in 16-bit format.
    
    :param input_image_path: Path to the input image file
    :param output_image_path: Path to save the interpolated output image
    :param scale_factor: Scaling factor for height (default: 2)
    """
    # Read the image in unchanged mode (keeps 16-bit depth if available)
    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    
    # Calculate new dimensions
    new_width = image.shape[1]
    new_height = int(image.shape[0] * scale_factor)
    
    # Resize the image using bicubic interpolation
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Save the output image in 16-bit format
    cv2.imwrite(output_image_path, resized_image)


def height_x2_bicubic(input_folder_path,output_folder_path ):
    # Process each image in the folder
    for img in tqdm(sorted(os.listdir(input_folder_path), key=extract_number)):
        input_image_path = os.path.join(input_folder_path, img)
        output_image_path = os.path.join(output_folder_path, img)
        bicubic_interpolate_image(input_image_path, output_image_path)





################################ rename files  ################################
def rename(src_path2, des_path2):
    fi_file_name=1
    for idx, files in enumerate(sorted(os.listdir(src_path2), key=extract_number5)):   #change the key, according to the scr path's folder name pattern 
        print("os.path.splitext(files)[0]",os.path.splitext(files)[0])
        # file_name_int = int(os.path.splitext(files)[0])
        
        ori_file_path = os.path.join(src_path2, files)
        new_name =f'{fi_file_name:08d}.png'
        dest_folder_path = os.path.join(dest_path2, new_name)
            # os.mkdir(dest_folder_path)
        shutil.copy(ori_file_path, dest_folder_path)
        fi_file_name+=1




########################################## 실행 ##########################################
# 이미지가 있는 폴더 경로
folder_path = '/mnt/sdb/VSR_Inference/bz_edit'
output_folder1 = '/mnt/sdb/VSR_Inference/bz_edit_slice'
os.makedirs(output_folder1, exist_ok=True)
slice_height(folder_path,output_folder1)

output_folder2 = '/mnt/sdb/VSR_Inference/bz_edit_slice_bicubic'
os.makedirs(output_folder2, exist_ok=True)
height_x2_bicubic(output_folder1, output_folder2)

output_folder3="/mnt/sdb/VSR_Inference/bz_edit_slice_bicubic_rename"
os.makedirs(output_folder3, exist_ok=True)
rename(output_folder2, output_folder3)
