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

import cv2
import os
from tqdm import tqdm

def extract_number(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        first_part = parts[0]
        second_part = os.path.splitext(parts[-1])[0]
        try:
            return int(first_part) * 100000 + int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)

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

input_folder_path = '/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice'
output_folder_path = '/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice_bicubic'

# Create output folder if it does not exist
os.makedirs(output_folder_path, exist_ok=True)

# Process each image in the folder
for img in tqdm(sorted(os.listdir(input_folder_path), key=extract_number)):
    input_image_path = os.path.join(input_folder_path, img)
    output_image_path = os.path.join(output_folder_path, img)
    bicubic_interpolate_image(input_image_path, output_image_path)
