import cv2
import os

# 이미지가 있는 폴더 경로
folder_path = '/mnt/sdb/VSR_Inference/dragon_full/ep001_16'

# 폴더 내 모든 이미지 파일 순회
for img_idx, filename in enumerate(sorted(os.listdir(folder_path))):
    image_path = os.path.join(folder_path, filename)
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"Failed to read {filename}. Skipping...")
        continue

    print(f"Processing {filename} with shape {image.shape}")

    processed_frame1 = image[::2, :, :]
    processed_frame2 = image[1::2, :, :]

    # BGR 이미지를 RGB로 변환 (Matplotlib에서 제대로 보이도록)
    processed_frame1_rgb = cv2.cvtColor(processed_frame1, cv2.COLOR_BGR2RGB)
    processed_frame2_rgb = cv2.cvtColor(processed_frame2, cv2.COLOR_BGR2RGB)

    # 결과 이미지 파일로 저장
    output_folder = '/mnt/sdb/VSR_Inference/dragon_full/ep001_16_slice'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame1_path = os.path.join(output_folder, f'{img_idx}_1.png')
    frame2_path = os.path.join(output_folder, f'{img_idx}_2.png')
    cv2.imwrite(frame1_path, cv2.cvtColor(processed_frame1_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(frame2_path, cv2.cvtColor(processed_frame2_rgb, cv2.COLOR_RGB2BGR))

    # print(f"Saved processed frames as {frame1_path} and {frame2_path}")

print("Processing complete.")