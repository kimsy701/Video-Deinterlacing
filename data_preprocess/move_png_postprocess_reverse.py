#move_png_postprocess.py의 역행 작업

import os
import shutil

# 폴더 경로와 범위 설정
base_path = "/media/user/T5 EVO/qtgmc_dataset/8월촬영본/0829/lr/pngdataset"  # 폴더가 위치한 기본 경로
# base_path = "/media/user/T5 EVO/qtgmc_dataset/8월촬영본/0829/gt/pngdataset"
start_folder = 768  # 시작 폴더 번호
end_folder = 888    # 끝 폴더 번호
n = 2  # 이동할 마지막 파일 개수

# 작업 수행
for folder in range(start_folder, end_folder):  
    current_folder_path = os.path.join(base_path, str(folder))
    print(current_folder_path)
    previous_folder_path = os.path.join(base_path, str(folder - 1))

    # 현재 폴더와 다음 폴더가 존재하는지 확인
    if os.path.exists(current_folder_path) and os.path.exists(previous_folder_path):
        # 현재 폴더의 파일 목록 가져오기
        files = sorted(os.listdir(current_folder_path))
        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg','.tiff'))]  # 이미지 파일 필터링

        if len(image_files) >= n:  # 파일이 n개 이상 있는 경우
            print("yes")
            # 마지막 n개 파일 선택
            # files_to_move = image_files[-n:]
            files_to_move = image_files[:n]

            # 파일 이동
            for file_name in files_to_move:
                print(file_name)
                source_path = os.path.join(current_folder_path, file_name)
                destination_path = os.path.join(previous_folder_path, file_name)
                shutil.move(source_path, destination_path)

