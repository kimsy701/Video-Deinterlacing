from PIL import Image
import os

def is_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))

def count_image_files(folder):
    count = 0
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) and is_image(filename):
            count += 1
    return count

def center_crop_and_save(input_folder, output_folder):
    target_size = (1920, 1080)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        num_image_files = count_image_files(root)

        if num_image_files == 3:
            for filename in files:
                filepath = os.path.join(root, filename)
                if is_image(filename):
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    output_path = os.path.join(output_subfolder, filename)

                    try:
                        with Image.open(filepath) as img:
                            img_width, img_height = img.size
                            left = (img_width - target_size[0]) // 2
                            top = (img_height - target_size[1]) // 2
                            right = left + target_size[0]
                            bottom = top + target_size[1]

                            cropped_img = img.crop((left, top, right, bottom))
                            if not os.path.exists(output_subfolder):
                                os.makedirs(output_subfolder)
                            cropped_img.save(output_path)

                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    input_folder = "/mnt/sda/danbi_triplet/sequences/0001"  # 입력 폴더 경로를 지정해주세요.
    output_folder = "/mnt/sda/db_triplet_1920"  # 결과 이미지를 저장할 폴더 경로를 지정해주세요.

    center_crop_and_save(input_folder, output_folder)
