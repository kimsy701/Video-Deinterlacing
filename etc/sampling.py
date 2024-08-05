import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
from PIL import Image
from tqdm import tqdm

##############################################  load every "im1.png" from each folder ############################################## 


def load_images_from_folders(base_path, base_path_im1):
    total_folders = 0
    images = []
    
    # Use os.scandir for efficient iteration
    with os.scandir(base_path) as entries:
        for entry in entries:
            if entry.is_dir():
                total_folders += 1
                folder_name = entry.name
                folder_path = entry.path
                image_path = os.path.join(folder_path, 'im1.png')
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    save_path = os.path.join(base_path_im1, f'{folder_name}_im1.png')
                    image.save(save_path)
    

############################################## Clustering ############################################## 

import os
import random
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import shutil

def organize_images(base_path, organized_base_path):  #21이라는 폴더에 21로 시작하는 이미지들, 38이라는 폴더에 38로 시작하는 이미지들 save 

    # Organize images into groups based on the prefix
    prefix = 21
    group_dir = os.path.join(organized_base_path, str(prefix))
    os.makedirs(group_dir, exist_ok=True)

    for entry in tqdm(sorted(os.scandir(base_path), key=lambda e: e.name)):

        if entry.name.split('_')[0] ==str(prefix):
            image_groups = {int(prefix): [] }
            image_groups[int(prefix)].append(entry.path)
            
            shutil.copy(entry, group_dir)
        else:
            prefix +=1
            group_dir = os.path.join(organized_base_path, str(prefix))
            os.makedirs(group_dir, exist_ok=True)
            shutil.copy(entry, group_dir)
            continue
        


def cluster_images(group_dir, cluster_rst_path, i, n_clusters=2): #10,30  # i=21,22,..
    images = [os.path.join(group_dir, f) for f in os.listdir(group_dir)]
    # print("images", images)
    image_data = []
    
    # Load images and prepare data for clustering
    for image_path in images:
        image = Image.open(image_path).convert('RGB')
        image_data.append(np.array(image).flatten())
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(image_data)
    
    # Organize images by cluster
    clustered_images = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(clusters):
        clustered_images[label].append(images[idx])
    
    # Sample images from each cluster
    for cluster, cluster_images in clustered_images.items():
        # print("cluster", cluster) 
        # print("cluster_images",cluster_images)
        cluster_path = os.path.join(cluster_rst_path, f'{i}_{cluster}')
        # print("cluster_path",cluster_path)
        os.makedirs(cluster_path, exist_ok=True)  # Create cluster result directory
        
        for img_path in cluster_images:
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(cluster_path, img_name))
    





def sample_process_images(base_path, i, n_clusters=2, n_samples=1):
    for c in range(n_clusters):
        # Construct the path to the directory
        directory_path = os.path.join(base_path, f'{i}_{c}')
        
        # List all files in the directory
        files = os.listdir(directory_path)
        # Randomly sample n_samples images
        sampled_files = random.sample(files, min(n_samples, len(files)))
        
        # Create a directory to store the sampled images
        sampled_directory_path = os.path.join(base_path, f'sampled_{i}_{c}')
        os.makedirs(sampled_directory_path, exist_ok=True)
        
        # Copy the sampled images to the new directory
        for file_name in sampled_files:
            src_path = os.path.join(directory_path, file_name)
            dst_path = os.path.join(sampled_directory_path, file_name)
            shutil.copyfile(src_path, dst_path)


def move_only_sampled_folder(cluster_and_sample_path, train_dataset_path, sampled_train_dataset_path):
    # Create the sampled_train_dataset directory if it does not exist
    if not os.path.exists(sampled_train_dataset_path):
        os.makedirs(sampled_train_dataset_path)

    # Iterate over each folder in the cluster_and_sample directory
    for folder_name in os.listdir(cluster_and_sample_path): #680개 (10class * 68)
        if folder_name.startswith("sampled_"):
            folder_path = os.path.join(cluster_and_sample_path, folder_name)
            # Process each image in the folder
            for image_name in os.listdir(folder_path):
                if image_name.endswith(".png"):
                    # Extract the first 6 digits from the image name
                    folder_identifier = "_".join(image_name.split("_")[:2])
                    
                    # Construct the path to the corresponding folder in train_dataset
                    source_folder_path = os.path.join(train_dataset_path, folder_identifier)
                    
                    # Construct the destination path
                    destination_folder_path = os.path.join(sampled_train_dataset_path, folder_identifier)
                    
                    # Copy the folder to the sampled_train_dataset directory
                    if not os.path.exists(destination_folder_path):
                        os.makedirs(destination_folder_path)
                    
                    # shutil.copytree(source_folder_path, destination_folder_path)
                    for file_name in os.listdir(source_folder_path):
                        source_file_path = os.path.join(source_folder_path, file_name)
                        destination_file_path = os.path.join(destination_folder_path, file_name)
                        shutil.copy2(source_file_path, destination_file_path)
        else:
            pass
                
                
        # # We only need to check one image per folder as all images in the folder have the same first 6 digits
        # break


def create_sample_gt(sampled_train_dataset_path, gt_dataset_path, sampled_gt_dataset_path):
    # Get the list of folder names from sampled_train_dataset
    sampled_folders = os.listdir(sampled_train_dataset_path)

    # Filter out only folders (exclude any files that might be in the directory)
    sampled_folders = [folder for folder in sampled_folders]

    # Iterate over the sampled folder names and copy them from gt_dataset to sampled_gt_dataset
    for folder_name in sampled_folders:
        src_folder = os.path.join(gt_dataset_path, folder_name)
        dest_folder = os.path.join(sampled_gt_dataset_path, folder_name)
        
        # Check if the folder exists in the gt_dataset
        # if os.path.exists(src_folder):
        #     # Copy the folder and its contents
        #     shutil.copytree(src_folder, dest_folder)
        #     print(f'Copied {src_folder} to {dest_folder}')
        # else:
        #     print(f'Folder {src_folder} does not exist in gt_dataset')
        shutil.copytree(src_folder, dest_folder)

    print("Copy operation completed.")



# Define paths
base_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/sampled_gt_frames_re'
base_path_im1 = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/sample_sampled/sampled_gt_frames_re_im1'
organized_base_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/sample_sampled/organize_by_prefix'
cluster_rst_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/sample_sampled/cluster_and_sample'
sampled_train_dataset_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/sample_sampled/sampled_sampled_gt_frames_re' #'/home/hy/sampled_gt_frames_re'
gt_dataset_path = '/home/kimsy701/deinter_datasets/gt_frames_re'
sampled_gt_dataset_path = '/home/kimsy701/deinter_datasets/sampled_gt_frames_re'

# Create necessary directories
if not os.path.exists(base_path_im1):
    os.makedirs(base_path_im1, exist_ok=True)
if not os.path.exists(organized_base_path):
    os.makedirs(organized_base_path, exist_ok=True)
if not os.path.exists(cluster_rst_path):
    os.makedirs(cluster_rst_path, exist_ok=True)
if not os.path.exists(sampled_train_dataset_path):
    os.makedirs(sampled_train_dataset_path, exist_ok=True)
if not os.path.exists(sampled_gt_dataset_path):
    os.makedirs(sampled_gt_dataset_path, exist_ok=True)



# Process images
#0
load_images_from_folders(base_path,base_path_im1)
# # 1
organize_images(base_path_im1, organized_base_path)
# # 2
for i in tqdm(range(21,89)):
    group_dir_fi = os.path.join(organized_base_path, str(i))
    cluster_images(group_dir_fi,cluster_rst_path, i, n_clusters=3) #각 i에 대해, 3개의 군집으로 나눔 (총 207개의 군집)
for i in tqdm(range(21,89)):
    sample_process_images(cluster_rst_path, i, n_clusters=3, n_samples=14) #각 군집에서 n_samples만큼 랜덤 추출
# 3
move_only_sampled_folder(cluster_rst_path, base_path, sampled_train_dataset_path)
# 4
create_sample_gt(sampled_train_dataset_path, gt_dataset_path, sampled_gt_dataset_path)
