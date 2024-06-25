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
def extract_number(directory):
    return int(directory)


# 각 폴더가 위치한 경로
source_path = "/home/kimsy701/deinter_dataset/interpolate_train_val_winter_rst"

# 이미지를 복사할 대상 폴더 경로
destination_folder = "/home/kimsy701/deinter_dataset/interpolate_train_val_winter_rst_fi"

if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)

# 이미지 번호 초기화
image_number = 1

# 각 폴더를 순회하며 이미지를 복사
for folder in sorted(os.listdir(source_path), key=extract_number):
    folder_path = os.path.join(source_path,folder)
    # 해당 폴더의 모든 파일을 순회
    for filename in sorted(os.listdir(folder_path)):
        # 원본 파일의 전체 경로
        source_file = os.path.join(folder_path, filename)
        # 새 파일 이름 생성
        new_filename = f'{image_number:08d}.png'
        # 새 파일의 전체 경로
        destination_file = os.path.join(destination_folder, new_filename)
        # 파일 복사
        shutil.copy(source_file, destination_file)
        # 이미지 번호 증가
        image_number += 1

print('이미지 복사가 완료되었습니다.')
