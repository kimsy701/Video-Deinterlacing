import os
import shutil
import glob
import cv2
import tqdm as tqdm


source_dir = '/mnt/sdd/4월촬영본'
output_image_dir = '/mnt/sdc/datasets/4월촬영본_onefolder' #-> 하는 중



# 모든 이미지를 모아둘 폴더가 없다면 생성
os.makedirs(output_image_dir, exist_ok=True)

def extract_number(directory):
    parts = directory.split('C')
    if len(parts) > 1:
        first_part =parts[0][-2] #parts[-1]
        second_part = parts[-1][1:3]
        try:
            return int(first_part)*1000 + int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)

def extract_number2(directory):

    return int(os.path.splitext(directory)[0])


def extract_number3(directory):
    parts = directory.split('_')
    if len(parts) > 1:
        first_part =parts[0]#parts[-1]
        second_part = parts[-1][1:]
        try:
            return int(first_part)*1000 + int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)

# 하위 폴더 안의 하위 폴더까지 탐색하며 이미지 복사
# image_counter = 1
# for root, dirs, files in sorted(os.walk(source_dir), key=lambda x: extract_number(x[0])):
#     print("root",root)
#     dirs.sort(key=extract_number2)
#     print("dirs",dirs)
#     for file in sorted(files):
#         print(file)
#         if file.endswith('.tiff'):
#             # 새로운 파일명 지정
#             new_filename = f"{str(image_counter).zfill(8)}.tiff"
#             source_file = os.path.join(root, file)
#             dest_file = os.path.join(output_image_dir, new_filename)
#             # 이미지 복사
#             shutil.copy(source_file, dest_file)
#             image_counter += 1
#             # print(f"Copied: {dest_file}")
            
            
# 하위 폴더 안의 하위 폴더까지 탐색하며 이미지 복사
image_counter = 1
for subfolder in sorted(os.listdir(source_dir), key=extract_number3):
    subfolder_path = os.path.join(source_dir, subfolder)
    # print(subfolder_path)
    for subsubfolder in tqdm.tqdm(sorted(os.listdir(subfolder_path), key=extract_number2)):
        subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
        for file in sorted(os.listdir(subsubfolder_path)):
            subsubfolderfile_path=os.path.join(subsubfolder_path, file)

            new_filename = f"{str(image_counter).zfill(8)}.png" #.tiff"
            dest_file = os.path.join(output_image_dir, new_filename)
            # 이미지 복사
            shutil.copy(subsubfolderfile_path, dest_file)
            image_counter += 1
            # print(f"Copied: {dest_file}")

# 이미지들을 하나의 영상으로 변환
#ffmpeg -framerate 29.97 -i %08d.tiff -vf "crop=720:480,scale=720:480:flags=bicubic" -c:v prores_ks -profile:v 4 -qscale:v 0 0829_video_bicubic.mov
#ffmpeg -framerate 29.97 -i %08d.tiff -vf "crop=720:480,scale=720:480:flags=lanczos" -c:v prores_ks -profile:v 4 -qscale:v 0 0829_video_lanczos3.mov

print("영상 변환이 완료되었습니다:", output_video_path)
