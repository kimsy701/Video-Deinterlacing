#4k dataset들을 50개의 frame들로 만들기
#원래 이미지 위치 : /mnt/nas3/8K 촬영본/8k_ren/A001_PNG
import os
import shutil

def slice_to_scene(src_path, dest_path):
    for subfolder in sorted(os.listdir(src_path)):
        subfolder_path = os.path.join(src_path, subfolder)
        images = sorted(os.listdir(subfolder_path))
        
        for i, img in enumerate(images):
            scene_number = i // 50
            new_folder_name = f"scene_{scene_number}"
            dest_folder_path = os.path.join(dest_path,subfolder, new_folder_name)
            
            # Ensure the destination directory exists
            os.makedirs(dest_folder_path, exist_ok=True)
            
            src_img_path = os.path.join(subfolder_path, img)
            dest_img_path = os.path.join(dest_folder_path, img)
            print("src_img_path:", src_img_path)
            print("dest_img_path:", dest_img_path)
            shutil.copy(src_img_path, dest_img_path)
            
def slice_to_scene2(src_path, dest_path):
    
        images = sorted(os.listdir(src_path))
        
        for i, img in enumerate(images):
            scene_number = i // 50
            new_folder_name = f"scene_{scene_number}"
            dest_folder_path = os.path.join(dest_path, new_folder_name)
            
            # Ensure the destination directory exists
            os.makedirs(dest_folder_path, exist_ok=True)
            
            src_img_path = os.path.join(src_path, img)
            dest_img_path = os.path.join(dest_folder_path, img)
            print("src_img_path:", src_img_path)
            print("dest_img_path:", dest_img_path)
            shutil.copy(src_img_path, dest_img_path)
            
                
            
# src_path = "/mnt/nas3/8K 촬영본/8k_ren/A001_PNG"
# dest_path = "/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset"
# src_path = "/mnt/nas3/8K 촬영본/8k_ren/A001_PNG/A001_C008_0527DM_001"
# dest_path = "/mnt/sda/8K 촬영본/8k_ren/A001_PNGdatset/A001_C008_0527DM_001"
# src_path = "/mnt/nas3/8K 촬영본/8k_ren/A001_PNG/A001_C009_05277P_001"
# dest_path = "/mnt/sda/8K 촬영본/8k_ren/A001_PNGdatset/A001_C009_05277P_001"
# src_path = "/mnt/nas3/8K 촬영본/8k_ren/A001_PNG/A001_C010_0527P3_001"
# dest_path = "/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset/A001_C010_0527P3_001"
src_path = "/mnt/nas3/8K 촬영본/8k_ren/A001_PNG/A001_C011_0527OW_001"
dest_path = "/mnt/nas2/8K 촬영본/8k_ren/A001_PNGdataset/A001_C011_0527OW_001"

# slice_to_scene(src_path,dest_path)
slice_to_scene2(src_path,dest_path)
