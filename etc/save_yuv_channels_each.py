import os
from PIL import Image
import numpy as np

def rgb_to_yuv(image):
    """Convert RGB image to YUV."""
    # RGB 이미지의 numpy 배열로 변환
    rgb = np.array(image)

    # YUV 변환 공식 적용
    yuv = np.empty(rgb.shape, dtype=np.float32)
    yuv[..., 0] = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # Y
    yuv[..., 1] = -0.14713 * rgb[..., 0] - 0.28886 * rgb[..., 1] + 0.436 * rgb[..., 2]  # U
    yuv[..., 2] = 0.615 * rgb[..., 0] - 0.51499 * rgb[..., 1] - 0.10001 * rgb[..., 2]  # V

    return yuv

def save_yuv_channels(idx, image_path, save_folder):
    # 이미지를 열고 RGB 채널을 분리합니다.
    image = Image.open(image_path).convert("RGB")
    
    # YUV로 변환
    yuv_image = rgb_to_yuv(image)

    # Y, U, V 채널로 분리
    y_channel = Image.fromarray(np.uint8(yuv_image[..., 0]), mode='L')  # Y 채널
    u_channel = Image.fromarray(np.uint8((yuv_image[..., 1] + 128)), mode='L')  # U 채널
    v_channel = Image.fromarray(np.uint8((yuv_image[..., 2] + 128)), mode='L')  # V 채널

    # 저장할 폴더가 없다면 생성합니다.
    os.makedirs(save_folder, exist_ok=True)

    # 각 채널을 이미지로 저장합니다.
    y_channel.save(os.path.join(save_folder, f'{idx+1:08d}.png'))
    # u_channel.save(os.path.join(save_folder, 'u_channel.png'))
    # v_channel.save(os.path.join(save_folder, 'v_channel.png'))


# 사용 예시
image_folder = '/mnt/sda/deinter_datasets/qtgmc_winter_rst_validdataset/qtgmc_winter_rst'  # 원본 이미지 경로
save_folder = '/mnt/sda/deinter_datasets/qtgmc_winter_rst_validdataset/qtgmc_winter_rst_y_ch'   # R, G, B 채널 저장할 폴더 경로

for idx, img in enumerate(sorted(os.listdir(image_folder))):
    image_path=os.path.join(image_folder, img)
    save_yuv_channels(idx, image_path, save_folder)

