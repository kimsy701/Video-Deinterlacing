import os
import shutil
# def extract_number(directory):
#     parts = directory.split('_')
#     if len(parts) > 1:
#         first_part =parts[0] #parts[-1]
#         second_part = parts[-1]
#         try:
#             return int(first_part)*1000 +int(second_part)
#         except ValueError:
#             return float('inf')  # Return infinity for non-numeric parts (optional)
#     return float('inf')  # Return infinity if no underscore is found (optional)
def extract_number2(directory):
    return int(directory)

def extract_number3(directory):
    parts = directory.split('/')[-1]
    parts_fi = parts.split('_')
    if len(parts_fi) > 1:
        second_part = parts_fi[1][3:] #original folder number
        third_part =parts_fi[2][-1] #subfolder number
        try:
            return int(second_part)*1000 +int(third_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)S
        
def extract_number3(directory):
    parts = os.path.splitext(directory)[0]
    return int(parts)


        

def copy_images(original_path, destination_path):
    # List all subfolders in the original path
   
    # Get list of images in subfolder and sort them
    images = sorted([img for img in os.listdir(original_path) if img.endswith('.png')], key=extract_number3)
    
    
    # Define the mapping of images to destination subfolders
    mapping = {}
    for j in range(len(images) - 1):
        folder_name = f'subfolder{j+1}' #vfi 결과물 : j+1 번째 이미지 
        indices = [j, j+2] if j+2 < len(images) else [j, len(images) - 1]
        mapping[folder_name] = indices
    
    # Copy images based on the defined mapping
    for folder_name, indices in mapping.items():
        destination_subfolder = os.path.join(destination_path, folder_name)
        os.makedirs(destination_subfolder, exist_ok=True)
        for index in indices:
            shutil.copy(os.path.join(original_path, images[index]), destination_subfolder)
            print(f"Copied {images[index]} to {destination_subfolder}")
                
                
def copy_images2(original_path, destination_path):
    # List all subfolders in the original path
    original_subfolders = sorted([f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))], key = lambda x: int(x))
    
    for i, subfolder in enumerate(original_subfolders):
        subfolder_path = os.path.join(original_path, subfolder)
        
        # Get list of images in subfolder and sort them
        images = sorted([img for img in os.listdir(subfolder_path) if img.endswith('.png')])
        
        
        # Define the mapping of images to destination subfolders
        mapping = {}
        for j in range(len(images) - 1):
            folder_name = f'ori_sub{i+1}_subfolder{j+1}'
            indices = [j, j+2] if j+2 < len(images) else [j, len(images) - 1]
            mapping[folder_name] = indices
        
        # Copy images based on the defined mapping
        for folder, indices in mapping.items():
            destination_subfolder = os.path.join(destination_path, folder)
            os.makedirs(destination_subfolder, exist_ok=True)
            for index in indices:
                shutil.copy(os.path.join(subfolder_path, images[index]), destination_subfolder)
                print(f"Copied {images[index]} to {destination_subfolder}")



def copy_first_and_last_image_of_subfolder(original_path, destination_path):
    # List all subfolders in the original path
    original_subfolders = sorted([f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))], key = lambda x: int(x))
    
    for i, subfolder in enumerate(original_subfolders):
        subfolder_path = os.path.join(original_path, subfolder)
        
    # Get list of images in subfolder and sort them
        images = sorted([img for img in os.listdir(subfolder_path) if img.endswith('.png')])
        
        # Ensure there are exactly 6 images
        if len(images) == 6:
            # Define the mapping of images to destination subfolders
            mapping = {
                f'ori_sub{i+1}_subfolder0': 0, #6개 중 2번째 이미지 
                f'ori_sub{i+1}_subfolder5': 5, #6개 중 5번째 이미지 
                # f'ori_sub{i+1}_subfolder1': [0, 2],
                # f'ori_sub{i+1}_subfolder2': [1, 3],
                # f'ori_sub{i+1}_subfolder3': [2, 4],
                # f'ori_sub{i+1}_subfolder4': [3, 5]
            }
            
            # Copy images based on the defined mapping
            for folder, index in mapping.items():
                destination_subfolder = os.path.join(destination_path, folder)
                os.makedirs(destination_subfolder, exist_ok=True)
                if index==0:
                    dest_file_name = '00000001.png'
                elif index==5:
                    dest_file_name = '00000006.png'
                
                destination_file_path=os.path.join(destination_subfolder, dest_file_name)
                shutil.copy(os.path.join(subfolder_path, images[index]), destination_file_path)
                print(f"Copied {images[index]} to {destination_subfolder}")
        
        
########## process ##########
        
# Example usage
original_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_toonefolder'
destination_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_input_re'
os.makedirs(destination_path, exist_ok = True)

copy_images(original_path, destination_path)
# copy_images2(original_path, destination_path)


            
# Example usage
original_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff'
destination_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_output_20240717_190636_2'
os.makedirs(destination_path, exist_ok = True)

# copy_first_and_last_image_of_subfolder(original_path, destination_path)
        
