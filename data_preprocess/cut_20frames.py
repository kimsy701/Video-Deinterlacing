import os
import shutil
import argparse

def copy_images_in_batches(src_path, dest_path, batch_size=20):
    # 이미지 파일 목록을 가져오기
    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    files.sort()  # 필요한 경우 정렬
    
    # batch 별로 파일을 복사
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        subfolder_path = os.path.join(dest_path, f'{i // batch_size + 1}')
        os.makedirs(subfolder_path, exist_ok=True)
        
        for file_name in batch:
            src_file = os.path.join(src_path, file_name)
            dest_file = os.path.join(subfolder_path, file_name)
            shutil.copy(src_file, dest_file)
    
    print(f"{len(files)}개의 파일이 {batch_size}개씩 나눠서 {dest_path}에 복사되었습니다.")



def main():
    parser = argparse.ArgumentParser(description="Process images in batches of 20.")
    parser.add_argument('--src_path', type=str, required=True, help='Path to the source folder containing images.')
    parser.add_argument('--dest_path', type=str, required=True, help='Path to save the processed images.')

    args = parser.parse_args()

    src_path = args.src_path
    dest_path = args.dest_path

    copy_images_in_batches(src_path, dest_path)

if __name__ == "__main__":
    main()
