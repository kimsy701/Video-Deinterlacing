import cv2
import os
import random

def extract_number1(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        second_part = parts[1]
        second_part_num_part = second_part[-3:]  # Take the last 3 characters of the second part
        
        try:
            return int(second_part_num_part)
        except ValueError:
            try:
                return float(second_part_num_part)
            except ValueError:
                return float('inf')  # Return infinity for non-numeric parts
        
    return float('inf')  # Return infinity if no underscore is found


def extract_number2(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        second_part = parts[1]
        
        try:
            return int(second_part)
        except ValueError:
            try:
                return float(second_part_num_part)
            except ValueError:
                return float('inf')  # Return infinity for non-numeric parts
        
    return float('inf')  # Return infinity if no underscore is found

def extract_number3(directory):

    try:
        # Splitting the filename by '.' to separate the numeric part from the extension
        parts = directory.split('.')
        if len(parts) > 1:
            numeric_part = parts[0]  # Take the first part before the dot
            # Convert the numeric part to integer
            return int(numeric_part)
        else:
            return float('inf')  # Return infinity if no extension found (optional)
    except ValueError:
        return float('inf')  # Return infinity if conversion fails


def spiral_crop(image, start_x, start_y, step_size=10):
    img_h, img_w, _ = image.shape
    
    # Initialize direction vectors: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_index = 0  # Start moving to the right
    
    # Initialize crop dimensions
    crop_width = 448  # Assuming a fixed crop size for simplicity
    crop_height = 256

    # Move to the next position in the current direction
    next_x = start_x + directions[direction_index][1] * step_size  # move right or left
    next_y = start_y + directions[direction_index][0] * step_size  # move down or up

    # Check if the next position is within bounds
    if (next_x < 0 or next_x + crop_width > img_w or
        next_y < 0 or next_y + crop_height > img_h):
        # If not, change direction (rotate clockwise)
        direction_index = (direction_index + 1) % 4
    else:
        start_x = next_x
        start_y = next_y
        
    if (0 <= start_x < img_w and 0 <= start_y < img_h and
            0 <= start_x + crop_width <= img_w and 0 <= start_y + crop_height <= img_h):
            cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
            return cropped_image, start_x, start_y


# Save the cropped images
ori_path = "/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset"
dest_path = "/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset_crop_spiral_10pixel"
os.makedirs(dest_path, exist_ok=True)

start_x = (3840 - 448) // 2  # Center x-coordinate
start_y = (2160 - 256) // 2  # Center y-coordinate

for largefolder in sorted(os.listdir(ori_path), key=extract_number1):
    largefolderpath = os.path.join(ori_path, largefolder)
    for scenefolder in sorted(os.listdir(largefolderpath), key=extract_number2):
        scenefolderpath = os.path.join(largefolderpath, scenefolder)
        
        for img_idx, img in enumerate(sorted(os.listdir(scenefolderpath), key=extract_number3)):
            img_path = os.path.join(scenefolderpath, img)
            print("img_path", img_path)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # Ensure the image is read correctly
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            
            cropped_image, start_x, start_y = spiral_crop(image, start_x, start_y, step_size=10)

            output_folder_path = os.path.join(dest_path, largefolder, scenefolder)
            try:
                os.makedirs(output_folder_path, exist_ok=True)
            except FileExistsError:
                pass  # Directory already exists, do nothing
            
            output_img_path = os.path.join(output_folder_path, f"{img_idx:08d}.png")
            cv2.imwrite(output_img_path, cropped_image)
