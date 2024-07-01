# pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
#mim install mmcv==1.7.2


import zipfile
import os
import shutil

def strip_leading_zeros(folder_name):
    # Remove leading zeros from the folder name
    return folder_name.lstrip('0')

def move_contents(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.move(s, d)
    os.rmdir(src)  # Remove the now-empty subfolder

################ unzip and make it to "21_1" format ################
def extract_and_rename(zip_file_path, destination_path):
    # Create a temporary directory for extraction
    temp_extract_path = os.path.join(destination_path, 'temp')
    os.makedirs(temp_extract_path, exist_ok=True)

    # Extract the zip file to the temporary directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # Process each main folder in the temporary extraction directory
    main_folders = [f for f in os.listdir(temp_extract_path) if os.path.isdir(os.path.join(temp_extract_path, f))]

    for main_folder in main_folders:
        main_folder_stripped = strip_leading_zeros(main_folder)
        main_folder_path = os.path.join(temp_extract_path, main_folder)
        
        # Get all subfolders in the main folder
        subfolders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
        
        for subfolder in subfolders:
            subfolder_stripped = strip_leading_zeros(subfolder)
            new_folder_name = f'{main_folder_stripped}_{subfolder_stripped}'
            new_folder_path = os.path.join(destination_path, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            old_folder_path = os.path.join(main_folder_path, subfolder)
            
            # Move contents of the subfolder to the new folder
            move_contents(old_folder_path, new_folder_path)

    # Remove the temporary extraction directory
    shutil.rmtree(temp_extract_path)

    print('Extraction and renaming completed successfully.')
    
"""
# Define the path to the zip file and the destination directory
zip_file_path = '/home/jovyan/deinter_datasets/gt_frames/vimeo_90k_4.zip'
destination_path = '/home/jovyan/deinter_datasets/gt_frames'

extract_and_rename(zip_file_path, destination_path)
"""



################ if already unzipped make it to 21_1, 21_2,...format (before : 00021 -> 0001,0002,0003,... ) ################

def reformat(start_path, destination_path):

    # Process each main folder in the temporary extraction directory
    main_folders = [f for f in os.listdir(start_path) if os.path.isdir(os.path.join(start_path, f))]

    for main_folder in main_folders:
        main_folder_stripped = strip_leading_zeros(main_folder)
        main_folder_path = os.path.join(start_path, main_folder)
        
        # Get all subfolders in the main folder
        subfolders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
        
        for subfolder in subfolders:
            subfolder_stripped = strip_leading_zeros(subfolder)
            new_folder_name = f'{main_folder_stripped}_{subfolder_stripped}'
            new_folder_path = os.path.join(destination_path, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            old_folder_path = os.path.join(main_folder_path, subfolder)
            
            # Move contents of the subfolder to the new folder
            move_contents(old_folder_path, new_folder_path)

            # Remove the temporary extraction directory
            # shutil.rmtree(old_folder_path)

    print('Extraction and renaming completed successfully.')

# Define the path and execute function
# start_path = '/home/kimsy701/deinter_datasets/gt_frames_re' 
# destination_path = '/home/kimsy701/deinter_datasets/gt_frames_re_fi'

# reformat(start_path, destination_path)


################ make 7 frames to 6 frames ################

import os

def seven_to_six(base_dir):
    # Iterate over each subdirectory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        # Check if the item is a directory
        if os.path.isdir(subdir_path):
            # List all files in the subdirectory../..
            files = os.listdir(subdir_path)
            # Check if 'im7' exists in the list of files
            if 'im7.png' in files:
                # Remove 'im7' from the list of files
                files.remove('im7.png')
                # Remove 'im7' file from the subdirectory
                os.remove(os.path.join(subdir_path, 'im7.png'))

# seven_to_six("/home/jovyan/deinter_datasets/gt_val")
# seven_to_six("/home/kimsy701/deinter_datasets/gt_frames_re") 
# seven_to_six("/home/jovyan/deinter_datasets/train_frame") ->완료
# seven_to_six("/home/jovyan/deinter_datasets/train_val") -> 완료

#최종 하위 폴더 확인 : gt_frames_re(64612), gt_val(7824), train_frame(64612), train_val(7824)
#최종 파일 개수 ( 6개 * 폴더 개수) 확인 :  gt_frames_re(387672), gt_val(46944), train_frame(387672), train_val(46944)

################ im1~im6외에 다른 이미는 삭제하는 코드 ################
import os

def delete_otherthan_imgs(base_path):

    # 유지할 파일 목록을 지정합니다.
    keep_files = {'im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png'}

    # 하위 폴더를 순회합니다.
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # 파일 이름이 유지할 파일 목록에 포함되어 있는지 확인합니다.
            if file not in keep_files:
                # 포함되어 있지 않으면 파일을 삭제합니다.
                file_path = os.path.join(root, file)
                print(f'Removing {file_path}')  # 삭제할 파일을 출력합니다.
                # os.remove(file_path)

# delete_otherthan_imgs('/home/jovyan/deinter_datasets/gt_frames_re')
