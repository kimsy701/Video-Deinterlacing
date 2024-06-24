###############  !!!!!!!!!!!!!!!!!!!   make moving video from 8k data (4k 4개로 자르되, stride씩 sliding하면서 4개의 인위적인 4k를 만듦)   !!!!!!!!!!!!!!!!!!!


import cv2
import os
import numpy as np
from tqdm import tqdm


########################### 각 영역마다 (1px, 1px) 씩 4k 이미지(3840 * 2160) 가 이동하는 프레임 생성 (방향: 각 영역에서 부터 가운데 방향으로, ex) top left 영역은 botton right  방향으로)###########################
def crop_shifted_images(image, top_left,top_right,bottom_left,bottom_right): 
    # print("image shape", image.shape) # (2160, 3840, 3)
    """Generate shifted images towards the center."""
    top_left_img=image[top_left[1]:top_left[3], top_left[0]:top_left[2], :] #(1080, 1920, 3)
    top_right_img=image[top_right[1]:top_right[3], top_right[0]:top_right[2], :]
    bottom_left_img=image[bottom_left[1]:bottom_left[3], bottom_left[0]:bottom_left[2], :]
    bottom_right_img=image[bottom_right[1]:bottom_right[3], bottom_right[0]:bottom_right[2], :]
    

    return top_left_img, top_right_img, bottom_left_img,bottom_right_img


###########################  8k data를 padding 영역 포함해서 겹치는 4개의 영역으로 나눠주기 ###########################
def process_images_in_folder(input_folder, output_folder, num_steps=50, interval=60, x_stride=1, y_stride=1):

    file_count = 0
    
    # First pass to count files
    with os.scandir(input_folder) as entries:
        for entry in entries:
            if entry.is_file():
                file_count += 1
    
    for i in tqdm(range(0,file_count,num_steps)):
        print("i",i)
        # if i % 50==0:
        sub_folder_idx = i // num_steps #50
        current_batch = []
        
        for region in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            fi_output_dir = os.path.join(output_folder, f"{sub_folder_idx}/8k_dataset_{region}/")
            print("fi_output_dir",fi_output_dir)
            os.makedirs(fi_output_dir, exist_ok=True, mode=0o7770)
        
        # Get the current batch of filenames #50개만 가져오기
        with os.scandir(input_folder) as entries:
            for entry in entries:
                if len(current_batch) < num_steps:
                    current_batch.append(entry.name)
                elif len(current_batch) >= num_steps:
                    break
                
        for j, filename in enumerate(current_batch):
            image_path = os.path.join(input_folder, filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            height, width = image.shape[:2]

            x_move = x_stride * i
            y_move = y_stride * i
            
            top_left = (x_move, y_move, x_move + width // 2, y_move + height // 2)
            top_right = (width - width // 2 - x_move, y_move, width - x_move, height // 2 + y_move)
            bottom_left = (x_move, height - height // 2 - y_move, width // 2 + x_move, height - y_move)
            bottom_right = (width - width // 2 - x_move, height - height // 2 - y_move, width - x_move, height - y_move)
            
            top_left_img, top_right_img, bottom_left_img, bottom_right_img = crop_shifted_images(image, top_left, top_right, bottom_left, bottom_right)

            for region, region_img in zip(
                ["top_left", "top_right", "bottom_left", "bottom_right"], 
                [top_left_img, top_right_img, bottom_left_img, bottom_right_img]
            ):
                fi_output_dir = os.path.join(output_folder, f"{sub_folder_idx}/8k_dataset_{region}/")
                output_path = os.path.join(fi_output_dir, f"{i + j - num_steps * sub_folder_idx:08d}.png")
                cv2.imwrite(output_path, region_img)
            
            
########################### still cut 인 부분만 frame들의 시작점 stride씩 움직이게 함 ###########################
def generate_start_frames(ranges):
    start_frames = []
    for start, end in ranges:
        current = start
        while current <= end:
            if current % 50 == 0:
                start_frames.append(current)
            current += 50
    return start_frames


# Define the still cut ranges
still_ranges = [
    (0, 3850),
    (4023, 5382),
    (5555, 5770),
    (5966, 6113),
    (6375, 6871),
    (7140, 7586),
    (7769, 7853),
    (8091, 10096)  # Assuming still cut continues indefinitely
]

def process_images_in_folder2(input_folder, output_folder, still_ranges, num_steps=50, interval=60, x_stride=1, y_stride=1):

    file_count = 0
    
    # First pass to count files
    with os.scandir(input_folder) as entries:
        for entry in entries:
            if entry.is_file():
                file_count += 1
    
    
    
    #still cut만 처리
    adjusted_ranges = []
    for start, end in still_ranges:
        adjusted_start = (start // 50) * 50
        adjusted_end = ((end // 50) + 1) * 50 - 1
        adjusted_ranges.append((adjusted_start, adjusted_end))

    # Generate the start frames
    start_50_frames = generate_start_frames(adjusted_ranges)
    print(start_50_frames)



    # for i in tqdm(range(0,file_count,num_steps)): #i=0,50,100,...
    for i in tqdm(start_50_frames):
        print("i",i)
        # if i % 50==0:
        sub_folder_idx = i // num_steps #50
        current_batch = []
        
        for region in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            fi_output_dir = os.path.join(output_folder, f"{sub_folder_idx}_{region}/")
            os.makedirs(fi_output_dir, exist_ok=True, mode=0o7770)
        
        # Get the current batch of filenames #50개만 가져오기
        # with os.scandir(input_folder) as entries:
        #     for entry in entries:
        #         if len(current_batch) < num_steps:
        #             current_batch.append(entry.name)
        #         elif len(current_batch) >= num_steps:
        #             break
        
        with os.scandir(input_folder) as entries:
            for entry in entries:
                if len(current_batch) < num_steps:
                    current_batch.append(entry.name)
                elif len(current_batch) >= num_steps:
                    break
                
        for j, filename in enumerate(current_batch): #50번 반복 
            image_path = os.path.join(input_folder, filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            height, width = image.shape[:2]

            x_move = x_stride #x_stride * i # *50,100,150,200,...
            y_move = y_stride #y_stride * i # *50,100,150,200,...
            
            top_left = (x_move+j, y_move+j, x_move + width // 2+j, y_move + height // 2+j)
            top_right = (width - width // 2 - x_move-j, y_move+j, width - x_move-j, height // 2 + y_move+j)
            bottom_left = (x_move+j, height - height // 2 - y_move-j, width // 2 + x_move+j, height - y_move-j)
            bottom_right = (width - width // 2 - x_move-j, height - height // 2 - y_move-j, width - x_move-j, height - y_move-j)
            
            top_left_img, top_right_img, bottom_left_img, bottom_right_img = crop_shifted_images(image, top_left, top_right, bottom_left, bottom_right)

            for region, region_img in zip(
                ["top_left", "top_right", "bottom_left", "bottom_right"], 
                [top_left_img, top_right_img, bottom_left_img, bottom_right_img]
            ):
                fi_output_dir = os.path.join(output_folder, f"{sub_folder_idx}_{region}/")
                output_path = os.path.join(fi_output_dir, f"{i + j - num_steps * sub_folder_idx:08d}.png") 
                cv2.imwrite(output_path, region_img)


####### Still cut들 따로 처리 ###### 

# Define the still cut ranges
still_ranges = [(0, 7980)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C002_0527WV"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C002_0527WV" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)




# Define the still cut ranges
still_ranges = [(0, 4590), (4896, 6442), (6651, 8316), (8561, 9772), (9884, 10388), (10550,11539)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C003_0527IL"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C003_0527IL" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)




# Define the still cut ranges
still_ranges = [(0, 3850),(4023, 5382),(5555, 5770),(5966, 6113),(6375, 6871),(7140, 7586),(7769, 7853),(8091, 10096)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C004_052761"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C004_052761" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)





# Define the still cut ranges
still_ranges = [(0, 455), (806, 1224), (1365, 2522), (2737, 2958), (3166, 4690), (4901, 5716), (5934, 6066), (6359, 7628), (7830, 8166), (8354, 14438)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C008_05272I"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C008_05272I" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)






# Define the still cut ranges
still_ranges = [(0, 2202), (2571, 5777), (6113, 7209)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C009_0527IY"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C009_0527IY" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)





# Define the still cut ranges
still_ranges = [(0, 1805), (2090, 3004)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C010_0527D8"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C010_0527D8" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)





# Define the still cut ranges
still_ranges = [(0, 2159), (2455, 3184), (3453, 4284), (4577, 5800)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C011_0527WX"  # Input folder path
output_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNGdataset/A003_C011_0527WX" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)
