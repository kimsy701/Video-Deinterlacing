import mmcv
import random 
import cv2
import numpy as np

def ChromaNoise(img, bit=255): 
    box_size = 50
    shift_range=10
    num_box=5
    img_h, img_w, _ = img.shape

    # Loop to apply 5 random chroma noise boxes
    for _ in range(num_box):
        print(_)
        # Select a random top-left corner for the b x b block
        top_left_x = random.randint(0, img_w - box_size)
        top_left_y = random.randint(0, img_h - box_size)
        # Define the bottom-right corner
        bottom_right_x = top_left_x + box_size
        bottom_right_y = top_left_y + box_size
        # Extract the region of interest (ROI)
        roi = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        print(top_left_x,top_left_y)
        # Split the ROI into B, G, R channels
        b, g, r = cv2.split(roi)

        # Apply random shift to B and G channels
        # b_shift = np.random.randint(-shift_range, shift_range)  # Random shift for blue channel
        # g_shift = np.random.randint(-shift_range, shift_range)  # Random shift for green channel
        b_shift=3
        g_shift=3
        
        # Shift blue and green channels horizontally (axis=1)
        b = np.roll(b, b_shift, axis=1)  # Shift blue channel
        g = np.roll(g, g_shift, axis=1)  # Shift green channel

        
        # Merge the shifted channels back together
        shifted_roi = cv2.merge((b, g, r))
        
        # Replace the region in the original image with the shifted ROI
        img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = shifted_roi
    return img


def ChromaNoise2(img, bit=255): 
    box_size = 100
    shift_range=5
    num_box=5
    img_h, img_w, _ = img.shape

    # Loop to apply 5 random chroma noise boxes
    for _ in range(num_box):
        # Select a random top-left corner for the b x b block
        top_left_x = random.randint(0, img_w - box_size)
        top_left_y = random.randint(0, img_h - box_size)
        
        # Define the bottom-right corner
        bottom_right_x = top_left_x + box_size
        bottom_right_y = top_left_y + box_size
        # Extract the region of interest (ROI)
        roi = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        print(top_left_x,top_left_y)

        # Split the ROI into B, G, R channels
        b, g, r = cv2.split(roi)

        # Apply random shift to B and G channels
        # b_shift = np.random.randint(-shift_range, shift_range)  # Random shift for blue channel
        # g_shift = np.random.randint(-shift_range, shift_range)  # Random shift for green channel

        
        # Shift blue and green channels
        b_shifted = np.roll(b, shift_range, axis=1)  # Shift blue channel
        g_shifted = np.roll(g, shift_range, axis=1)  # Shift green channel
        

        # Merge the shifted channels back together
        shifted_roi = cv2.merge((b_shifted, g_shifted, r))
        
        # Step 1: Slice the left 10 columns
        left_cols = shifted_roi[:, :shift_range]

        # Step 2: Slice the remaining columns
        remaining_cols = shifted_roi[:, shift_range:]

        # Step 3: Concatenate the remaining columns with the left 10 columns added to the right
        shifted_roi_with_wrap = np.concatenate((remaining_cols, left_cols), axis=1)
        
        # Replace the region in the original image, adjusting the x-coordinate
        # img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = shifted_roi
        # img[top_left_y:bottom_right_y, top_left_x  :bottom_right_x ] = shifted_roi_with_wrap
        img[top_left_y:bottom_right_y, top_left_x  :bottom_right_x- shift_range] = remaining_cols

    return img

def diversify_chroma_channel(channel, target_std_multiplier=2.0):
    # Calculate the current mean and standard deviation
    mean = np.mean(channel)
    std = np.std(channel)
    
    # Generate random values with the same mean and increased variance
    new_std = std * target_std_multiplier  # Increase standard deviation
    random_values = np.random.normal(loc=mean, scale=new_std, size=channel.shape)

    # Clip the new values to ensure they are in the valid range [0, 255]
    diversified_channel = np.clip(random_values, 0, 255).astype(np.uint8)  # Convert to uint8

    return diversified_channel

def add_random_variation(channel, variation_range=10):
    # Create a unique random variation for each pixel in the channel
    random_variation = np.random.randint(-variation_range, variation_range + 1, channel.shape)
    channel = np.clip(channel + random_variation, 0, 255)  # Ensure values are in [0, 255]
    return channel.astype(channel.dtype)

def ChromaNoiseYUV(img, bit=255): 
    box_size = 100
    num_box = 5
    img_h, img_w, _ = img.shape

    # Loop to apply 5 random chroma noise boxes
    for _ in range(num_box):
        # Select a random top-left corner for the box
        # top_left_x = random.randint(0, img_w - box_size)
        # top_left_y = random.randint(0, img_h - box_size)
        
        top_left_x = int(img_w/2)
        top_left_y = int(img_h/2)
        
        # Define the bottom-right corner
        bottom_right_x = top_left_x + box_size
        bottom_right_y = top_left_y + box_size
        
        # Extract the region of interest (ROI)
        roi = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        print(top_left_x, top_left_y)

        # Convert ROI from BGR to YUV
        yuv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        
        # Split the ROI into Y, U, V channels
        y, u, v = cv2.split(yuv_roi)

        # Diversify U and V channels
        # u_diversified = diversify_chroma_channel(u, scale=random.uniform(1.2, 1.5))  # Scale U channel
        # v_diversified = diversify_chroma_channel(v, scale=random.uniform(1.2, 1.5))  # Scale V channel
        
        u_diversified = diversify_chroma_channel(u, target_std_multiplier=1.3)  # Scale U channel
        v_diversified = diversify_chroma_channel(v, target_std_multiplier=1.3)

        # Add random variations to U and V channels
        u_final = add_random_variation(u_diversified, variation_range=5)
        v_final = add_random_variation(v_diversified, variation_range=5)

        # Merge the diversified channels back together

        diversified_yuv_roi = cv2.merge((y.astype(np.uint8), u_final.astype(np.uint8), v_final.astype(np.uint8)))

        # Convert the diversified YUV ROI back to BGR
        diversified_roi = cv2.cvtColor(diversified_yuv_roi, cv2.COLOR_YUV2BGR)

        # Replace the region in the original image with the new diversified ROI
        img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = diversified_roi

    return img

def diversify_chroma_channel(channel, target_std_multiplier=1.3):
    """
    Expand the distribution of the chroma channel (U or V) by scaling it to match a larger target standard deviation.
    """
    mean, std = np.mean(channel), np.std(channel)
    target_std = std * target_std_multiplier
    expanded_channel = (channel - mean) * (target_std / std) + mean
    return expanded_channel

def remap_to_original_range(channel, original_channel):
    """
    Remap the expanded channel values to the range of the original channel values.
    """
    original_min, original_max = np.min(original_channel), np.max(original_channel)
    remapped_channel = np.clip(channel, original_min, original_max)
    return remapped_channel

def ChromaNoiseYUV_Modified(img, bit=255, shift_range=5): 
    box_size = 300
    num_box = 5
    img_h, img_w, _ = img.shape

    # Loop to apply 5 random chroma noise boxes
    for _ in range(num_box):
        # Select a random top-left corner for the box
        top_left_x = random.randint(0, img_w - box_size)
        top_left_y = random.randint(0, img_h - box_size)
        
        # Define the bottom-right corner
        bottom_right_x = top_left_x + box_size
        bottom_right_y = top_left_y + box_size
        
        # Extract the region of interest (ROI)
        roi = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        print(top_left_x, top_left_y)
        
        # Convert ROI from BGR to YUV
        yuv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        
        # Split the ROI into Y, U, V channels
        y, u, v = cv2.split(yuv_roi)
        diversified_yuv_roi = cv2.merge((y, u, v))
        diversified_bgr_roi = cv2.cvtColor(diversified_yuv_roi, cv2.COLOR_YUV2BGR)
        cv2.imwrite("/home/spocklabs/hy/inshorts_vsr/utils/diversified_yuv_roi.png",diversified_bgr_roi)

        # Diversify U and V channels by expanding their distributions
        u_expanded = diversify_chroma_channel(u, target_std_multiplier=5)
        v_expanded = diversify_chroma_channel(v, target_std_multiplier=5)

        # Remap the expanded U and V channels back to the original range
        u_final = remap_to_original_range(u_expanded, u)
        v_final = remap_to_original_range(v_expanded, v)
        
        # Apply np.roll for shifting U and V channels
        # u_rolled = np.roll(u_final, random.randint(-shift_range, shift_range), axis=1)
        # v_rolled = np.roll(v_final, random.randint(-shift_range, shift_range), axis=1)
        u_rolled = np.roll(u_final, shift_range, axis=1)
        v_rolled = np.roll(v_final, shift_range, axis=1)

        # Merge the rolled and remapped U, V channels back with Y channel
        diversified_yuv_roi = cv2.merge((y.astype(np.uint8), u_rolled.astype(np.uint8), v_rolled.astype(np.uint8)))
        diversified_bgr_roi = cv2.cvtColor(diversified_yuv_roi, cv2.COLOR_YUV2BGR)


        # Step 1: Slice the left 10 columns
        left_cols = diversified_yuv_roi[:, :shift_range]

        # Step 2: Slice the remaining columns
        remaining_cols = diversified_yuv_roi[:, shift_range:]
        remaining_cols_bgr = cv2.cvtColor(remaining_cols, cv2.COLOR_YUV2BGR)
     
        # Step 3: Concatenate the remaining columns with the left 10 columns added to the right
        shifted_roi_with_wrap = np.concatenate((remaining_cols, left_cols), axis=1)
        shifted_roi_with_wrap_bgr = cv2.cvtColor(shifted_roi_with_wrap, cv2.COLOR_YUV2BGR)

        # Replace the region in the original image, adjusting the x-coordinate
        # img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = shifted_roi
        # img[top_left_y:bottom_right_y, top_left_x  :bottom_right_x ] = shifted_roi_with_wrap_bgr
        img[top_left_y:bottom_right_y, top_left_x+ shift_range :bottom_right_x] = remaining_cols_bgr

    return img

def ChromaNoiseYUV_GlobalModified(img, bit=255, shift_range=5): 
    num_box = 5
    img_h, img_w, _ = img.shape

    # Loop to apply 5 random chroma noise boxes

    
    # Extract the region of interest (ROI)
    roi = img
    
    # Convert ROI from BGR to YUV
    yuv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
    
    # Split the ROI into Y, U, V channels
    y, u, v = cv2.split(yuv_roi)
    diversified_yuv_roi = cv2.merge((y, u, v))

    # Diversify U and V channels by expanding their distributions
    u_expanded = diversify_chroma_channel(u, target_std_multiplier=5)
    v_expanded = diversify_chroma_channel(v, target_std_multiplier=5)

    # Remap the expanded U and V channels back to the original range
    u_final = remap_to_original_range(u_expanded, u)
    v_final = remap_to_original_range(v_expanded, v)
    
    # Apply np.roll for shifting U and V channels
    # u_rolled = np.roll(u_final, random.randint(-shift_range, shift_range), axis=1)
    # v_rolled = np.roll(v_final, random.randint(-shift_range, shift_range), axis=1)
    u_rolled = np.roll(u_final, shift_range, axis=1)
    v_rolled = np.roll(v_final, shift_range, axis=1)

    # Merge the rolled and remapped U, V channels back with Y channel
    diversified_yuv_roi = cv2.merge((y.astype(np.uint8), u_rolled.astype(np.uint8), v_rolled.astype(np.uint8)))
    diversified_bgr_roi = cv2.cvtColor(diversified_yuv_roi, cv2.COLOR_YUV2BGR)


    # Step 1: Slice the left 10 columns
    left_cols = diversified_yuv_roi[:, :shift_range]

    # Step 2: Slice the remaining columns
    remaining_cols = diversified_yuv_roi[:, shift_range:]
    remaining_cols_bgr = cv2.cvtColor(remaining_cols, cv2.COLOR_YUV2BGR)
    
    # Step 3: Concatenate the remaining columns with the left 10 columns added to the right
    shifted_roi_with_wrap = np.concatenate((remaining_cols, left_cols), axis=1)
    shifted_roi_with_wrap_bgr = cv2.cvtColor(shifted_roi_with_wrap, cv2.COLOR_YUV2BGR)

    # Replace the region in the original image, adjusting the x-coordinate
    # img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = shifted_roi
    # img[top_left_y:bottom_right_y, top_left_x  :bottom_right_x ] = shifted_roi_with_wrap_bgr
    img[:, shift_range :] = remaining_cols_bgr

    return img

img=cv2.imread("/home/spocklabs/hy/inshorts_vsr/utils/x4-x200001498.png")
chroma_output = ChromaNoiseYUV_GlobalModified(img,shift_range=10)
cv2.imwrite("/home/spocklabs/hy/inshorts_vsr/utils/x4-x200001498_cnoise_yuv.png",chroma_output)


# img=cv2.imread("/home/spocklabs/hy/inshorts_vsr/utils/x4-x200001498.png")
# chroma_output = ChromaNoise2(img)
# cv2.imwrite("/home/spocklabs/hy/inshorts_vsr/utils/x4-x200001498_cnoise.png",chroma_output)


import os
from PIL import Image

def rgb_to_yuv(image):
    """Convert RGB image to YUV."""
    # RGB 이미지의 numpy 배열로 변환
    rgb = np.array(image)

    # YUV 변환 공식 적용
    yuv = np.empty(rgb.shape, dtype=np.float32)
    yuv[..., 0] = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # Y
    yuv[..., 1] = -0.14713 * rgb[..., 0] - 0.28886 * rgb[..., 1] + 0.436 * rgb[..., 2]  # U
    yuv[..., 2] = 0.615 * rgb[..., 0] - 0.51499 * rgb[..., 1] - 0.10001 * rgb[..., 2]  # V

    return yuv

def save_yuv_channels(image_path, save_folder):
    # 이미지를 열고 RGB 채널을 분리합니다.
    image = Image.open(image_path).convert("RGB")
    
    # YUV로 변환
    yuv_image = rgb_to_yuv(image)

    # Y, U, V 채널로 분리
    y_channel = Image.fromarray(np.uint8(yuv_image[..., 0]), mode='L')  # Y 채널
    u_channel = Image.fromarray(np.uint8((yuv_image[..., 1] + 128)), mode='L')  # U 채널
    v_channel = Image.fromarray(np.uint8((yuv_image[..., 2] + 128)), mode='L')  # V 채널

    # 저장할 폴더가 없다면 생성합니다.
    os.makedirs(save_folder, exist_ok=True)

    # 각 채널을 이미지로 저장합니다.
    y_channel.save(os.path.join(save_folder, 'y_channel.png'))
    u_channel.save(os.path.join(save_folder, 'u_channel.png'))
    v_channel.save(os.path.join(save_folder, 'v_channel.png'))


# 사용 예시
image_path = '/home/spocklabs/hy/inshorts_vsr/utils/x4-x200001498_cnoise_yuv.png'  # 원본 이미지 경로
save_folder = '/home/spocklabs/hy/inshorts_vsr/utils/channels'   # R, G, B 채널 저장할 폴더 경로
save_yuv_channels(image_path, save_folder)

image_path = '/home/spocklabs/hy/inshorts_vsr/utils/ori_bicubic_00000020.png'  # 원본 이미지 경로
save_folder = '/home/spocklabs/hy/inshorts_vsr/utils/channels_noise'   # R, G, B 채널 저장할 폴더 경로
save_yuv_channels(image_path, save_folder)
