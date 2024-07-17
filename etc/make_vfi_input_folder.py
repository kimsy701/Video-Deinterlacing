import os
import shutil

def copy_images(original_path, destination_path):
    # List all subfolders in the original path
    original_subfolders = sorted([f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))])
    
    for i, subfolder in enumerate(original_subfolders):
        subfolder_path = os.path.join(original_path, subfolder)
        
        # Get list of images in subfolder and sort them
        images = sorted([img for img in os.listdir(subfolder_path) if img.endswith('.png')])
        
        # Ensure there are exactly 6 images
        if len(images) == 6:
            # Define the mapping of images to destination subfolders
            mapping = {
                f'ori_sub{i+1}_subfolder1': [0, 2],
                f'ori_sub{i+1}_subfolder2': [1, 3],
                f'ori_sub{i+1}_subfolder3': [2, 4],
                f'ori_sub{i+1}_subfolder4': [3, 5]
            }
            
            # Copy images based on the defined mapping
            for folder, indices in mapping.items():
                destination_subfolder = os.path.join(destination_path, folder)
                os.makedirs(destination_subfolder, exist_ok=True)
                for index in indices:
                    shutil.copy(os.path.join(subfolder_path, images[index]), destination_subfolder)
                    print(f"Copied {images[index]} to {destination_subfolder}")

# Example usage
original_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff'
destination_path = '/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_input'
os.makedirs(destination_path, exist_ok = True)

copy_images(original_path, destination_path)
