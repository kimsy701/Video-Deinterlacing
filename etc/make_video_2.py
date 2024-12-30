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


def generate_start_frames_folders(ranges):
    start_frames_folders = []
    for start, end in ranges:
        current = start
        while current <= end:
            if current % 50 == 0:
                start_frames_folders.append(current//50)
            current += 50
    return start_frames_folders




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
   
    start_50_folders = generate_start_frames_folders(adjusted_ranges)



    # for i in tqdm(range(0,file_count,num_steps)): #i=0,50,100,...
    for i in tqdm(start_50_frames):
        print("i",i)
        # if i % 50==0:
        sub_folder_idx = i // num_steps #50
        current_batch = []
        
        for region in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            fi_output_dir = os.path.join(output_folder, f"{sub_folder_idx}_{region}/")
            os.makedirs(fi_output_dir, exist_ok=True, mode=0o7770)
        

        current_batch = []
        
        with os.scandir(input_folder) as entries:
            for ori_file_idx, entry in enumerate(entries):
                if ori_file_idx >= i and ori_file_idx < i + 50:
                    if entry.is_file():
                        current_batch.append(entry.name)
            
                
        for j, filename in enumerate(current_batch): #50번 반복 
            if i==1450:
                print("j",j)
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
# still_ranges = [(0, 7980)]

# input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C002_0527WV"  # Input folder path
# output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C002_0527WV" #/A003_C002_0527WV

# process_images_in_folder(input_folder, output_folder)
# process_images_in_folder2(input_folder, output_folder, still_ranges)
"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5750, 5800, 5850, 5900, 5950, 6000, 6050, 6100, 6150, 6200, 6250, 6300, 6350, 6400, 6450, 6500, 6550, 6600, 6650, 6700, 6750, 6800, 6850, 6900, 6950, 7000, 7050, 7100, 7150, 7200, 7250, 7300, 7350, 7400, 7450, 7500, 7550, 7600, 7650, 7700, 7750, 7800, 7850, 7900, 7950]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 
141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]
"""




# # Define the still cut ranges
# still_ranges = [(0, 4590), (4896, 6442), (6651, 8316), (8561, 9772), (9884, 10388), (10550,11500)] #(10550,11539)

# input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C003_0527IL"  # Input folder path
# output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C003_0527IL" #/A003_C002_0527WV

# # process_images_in_folder(input_folder, output_folder)
# process_images_in_folder2(input_folder, output_folder, still_ranges)
"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5750, 5800, 5850, 5900, 5950, 6000, 6050, 6100, 6150, 6200, 6250, 6300, 6350, 6400, 6650, 6700, 6750, 6800, 6850, 6900, 6950, 7000, 7050, 7100, 7150, 7200, 7250, 7300, 7350, 7400, 7450, 7500, 7550, 7600, 7650, 7700, 7750, 7800, 7850, 7900, 7950, 8000, 8050, 8100, 8150, 8200, 8250, 8300, 8550, 8600, 8650, 8700, 8750, 8800, 8850, 8900, 8950, 9000, 9050, 9100, 9150, 9200, 9250, 9300, 9350, 9400, 9450, 9500, 9550, 9600, 9650, 9700, 9750, 9850, 9900, 9950, 10000, 10050, 10100, 10150, 10200, 10250, 10300, 10350, 10550, 10600, 10650, 10700, 10750, 10800, 10850, 10900, 10950, 11000, 11050, 11100, 11150, 11200, 11250, 11300, 11350, 11400, 11450, 11500]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 97, 98, 99, 100,
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 133, 134, 135, 136, 137, 138, 139, 140, 
141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 
221, 222, 223, 224, 225, 226, 227, 228, 229, 230]
"""




# # Define the still cut ranges
# still_ranges = [(0, 3850),(4023, 5382),(5555, 5770),(5966, 6113),(6375, 6871),(7140, 7586),(7769, 7853),(8091, 10050)] #(8091, 10096)

# input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C004_052761"  # Input folder path
# output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C004_052761" #/A003_C002_0527WV

# # process_images_in_folder(input_folder, output_folder)
# process_images_in_folder2(input_folder, output_folder, still_ranges)
"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5550, 5600, 5650, 5700, 5750, 5950, 6000, 6050, 6100, 6350, 6400, 6450, 6500, 6550, 6600, 6650, 6700, 6750, 6800, 6850, 7100, 7150, 7200, 7250, 7300, 7350, 7400, 7450, 7500, 7550, 7750, 7800, 7850, 8050, 8100, 8150, 8200, 8250, 8300, 8350, 8400, 8450, 8500, 8550, 8600, 8650, 8700, 8750, 8800, 8850, 8900, 8950, 9000, 9050, 9100, 9150, 9200, 9250, 9300, 9350, 9400, 9450, 9500, 9550, 9600, 9650, 9700, 9750, 9800, 9850, 9900, 9950, 10000, 10050]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 
101, 102, 103, 104, 105, 106, 107, 111, 112, 113, 114, 115, 119, 120, 121, 122, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 
142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 155, 156, 157, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201]
"""





# # Define the still cut ranges
# still_ranges = [(0, 455), (806, 1224), (1365, 2522), (2737, 2958), (3166, 4690), (4901, 5716), (5934, 6066), (6359, 7628), (7830, 8166), (8354, 14400)] #(8354, 14438)

# input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C008_05272I"  # Input folder path 
# output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C008_05272I" #/A003_C002_0527WV

# # process_images_in_folder(input_folder, output_folder)
# process_images_in_folder2(input_folder, output_folder, still_ranges)

# i:1450, j:13에서 이미지 none이라고 뜸..


"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2700, 2750, 2800, 2850, 2900, 2950, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5900, 5950, 6000, 6050, 6350, 6400, 6450, 6500, 6550, 6600, 6650, 6700, 6750, 6800, 6850, 6900, 6950, 7000, 7050, 7100, 7150, 7200, 7250, 7300, 7350, 7400, 7450, 7500, 7550, 7600, 7800, 7850, 7900, 7950, 8000, 8050, 8100, 8150, 8350, 8400, 8450, 8500, 8550, 8600, 8650, 8700, 8750, 8800, 8850, 8900, 8950, 9000, 9050, 9100, 9150, 9200, 9250, 9300, 9350, 9400, 9450, 9500, 9550, 9600, 9650, 9700, 9750, 9800, 9850, 9900, 9950, 10000, 10050, 10100, 10150, 10200, 10250, 10300, 10350, 10400, 10450, 10500, 10550, 10600, 10650, 10700, 10750, 10800, 10850, 10900, 10950, 11000, 11050, 11100, 11150, 11200, 11250, 11300, 11350, 11400, 11450, 11500, 11550, 11600, 11650, 11700, 11750, 11800, 11850, 11900, 11950, 12000, 12050, 12100, 12150, 12200, 12250, 12300, 12350, 12400, 12450, 12500, 12550, 12600, 12650, 12700, 12750, 12800, 12850, 12900, 12950, 13000, 13050, 13100, 13150, 13200, 13250, 13300, 13350, 13400, 13450, 13500, 13550, 13600, 13650, 13700, 13750, 13800, 13850, 13900, 13950, 14000, 14050, 14100, 14150, 14200, 14250, 14300, 14350, 14400]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 98, 99, 100, 
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 118, 119, 120, 121, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 156, 157, 158, 159, 160, 161, 162, 163, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 
181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 
221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 
261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288]
"""





# # Define the still cut ranges
# still_ranges = [(0, 2202), (2571, 5777), (6113, 7200)] #(6113, 7209)

# input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C009_0527IY"  # Input folder path
# output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C009_0527IY" #/A003_C002_0527WV

# # process_images_in_folder(input_folder, output_folder)
# process_images_in_folder2(input_folder, output_folder, still_ranges)
"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5750, 6100, 6150, 6200, 6250, 6300, 6350, 6400, 6450, 6500, 6550, 6600, 6650, 6700, 6750, 6800, 6850, 6900, 6950, 7000, 7050, 7100, 7150, 7200]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144]
"""




# # Define the still cut ranges
# still_ranges = [(0, 1805), (2090, 3000)] #(2090, 3004)

# input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C010_0527D8"  # Input folder path
# output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C010_0527D8" #/A003_C002_0527WV

# # process_images_in_folder(input_folder, output_folder)
# process_images_in_folder2(input_folder, output_folder, still_ranges)
"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
"""



# # Define the still cut ranges
still_ranges = [(0, 2159), (2455, 3184), (3453, 4284), (4577, 5800)]

input_folder = "/mnt/nas3/8K 촬영본/8k_ren/A003_PNG/A003_C011_0527WX"  # Input folder path
output_folder = "/mnt/nas2/8K 촬영본/8k_ren/A003_PNGdataset/A003_C011_0527WX" #/A003_C002_0527WV

# # process_images_in_folder(input_folder, output_folder)
process_images_in_folder2(input_folder, output_folder, still_ranges)
"""
[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5750, 5800]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 49, 50, 
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
"""
