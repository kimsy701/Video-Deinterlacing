import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import torchvision

def sobel_edge_score(image, threshold=100): #200~270
    # Check if image is loaded correctly
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Apply Sobel operator in the x direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Apply threshold to keep only significant gradients
    sobel_magnitude = np.where(sobel_magnitude > threshold, sobel_magnitude, 0)

    
    # Calculate the sum of significant gradient magnitudes
    edge_score = np.sum(sobel_magnitude)
    
    # Normalize by the number of pixels to prevent size bias
    normalized_edge_score = edge_score / image.size
    
    return normalized_edge_score, sobel_magnitude

def calculate_high_frequency_energy(hf_image):
    # 고주파 성분의 절대값을 구하여 에너지를 계산
    hf_energy = np.sum(np.abs(hf_image))
    return hf_energy

class LowFreqFilter(nn.Module):
    def __init__(self):
        super(LowFreqFilter, self).__init__()
        self.lowpass_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        # Manually set the weights
        weights = torch.tensor([
            [0.1, 0.2, 0.1],
            [0.2, 0.3, 0.2],
            [0.1, 0.2, 0.1]
        ])

        with torch.no_grad():
            self.lowpass_filter.weight.copy_(weights.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x):
        #그냥 weight
        # x_lowfreq = self.lowpass_filter(x)
        #가우시안 blur
        blurrer = torchvision.transforms.GaussianBlur(111,sigma=(100,200))
        x_lowfreq = blurrer(x)
        x_highfreq = x - x_lowfreq
        print(torch.mean(x), torch.mean(x_lowfreq), torch.mean(x_highfreq))
        return x_lowfreq, x_highfreq

def find_HF_LF(image):
    # Convert the image to a PyTorch tensor and add a batch dimension
    image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
    
    # Initialize the low-frequency filter
    lowfreqfilter = LowFreqFilter()
    
    # Apply the filter to the image tensor
    lf_img, hf_img = lowfreqfilter(image_tensor)
    # print(hf_img.shape) #torch.Size([1, 1, 3744, 5616])
    
    # Convert the low and high-frequency components back to NumPy arrays
    lf_img_np = lf_img.squeeze(0).squeeze(0).detach().cpu().numpy() * 255.0
    hf_img_np = hf_img.squeeze(0).squeeze(0).detach().cpu().numpy() * 255.0
    
    lf_img_np = np.clip(lf_img_np,0,255)
    hf_img_np = np.clip(hf_img_np,0,255)
    
    return lf_img_np.astype(np.uint8), hf_img_np.astype(np.uint8)

def calculate_sharpness(image):
    # Convert to grayscale
    
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    print("laplacian",laplacian)
    
    # Calculate the variance of the Laplacian (sharpness measure)
    sharpness = laplacian.var()
    
    return sharpness

def create_lf_mask(lf_image, threshold=100):
    lr_sharpness = calculate_sharpness(lf_image)
    lf_mask = (lr_sharpness < threshold).astype(np.uint8)
    print(lf_mask)
    return lf_mask 


######################################    Applying part     ######################################
#### path ####
origin_path = "/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup/모래시계6회_초반11분_QTGMCeven_sample/00000450.png"
# origin_path_gray_path="/mnt/sdb/VSR_Inference/pexels-mccutcheon-1148998_gray.png"
origin_lfhf="/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup_reduce4.5/ori_lfhf"
mask_dir="/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup_reduce4.5/mask_lfhf"

#### Apply Sobel edge score to 전체 image #### 
#original 이미지 처리
origin_image = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
# cv2.imwrite(origin_path_gray_path, origin_image)
# HF/LF 이미지 분리
lf_ori_image, hf_ori_image = find_HF_LF(origin_image)
os.makedirs(origin_lfhf, exist_ok=True)
lf_ori_image_dir=os.path.join(origin_lfhf, "lf.png")
hf_ori_image_dir=os.path.join(origin_lfhf, "hf.png")
cv2.imwrite(lf_ori_image_dir, lf_ori_image)
cv2.imwrite(hf_ori_image_dir, hf_ori_image) 

#lf_mask, hf_mask 계산
lf_mask = create_lf_mask(lf_ori_image)
hf_mask = 1-lf_mask
os.makedirs(mask_dir, exist_ok=True)
lf_mask_dir=os.path.join(mask_dir, "lfmask.png")
hf_mask_dir=os.path.join(mask_dir, "hfmask.png")
print(type(lf_mask))
cv2.imwrite(lf_mask_dir, lf_mask*255)
cv2.imwrite(hf_mask_dir, hf_mask*255) 

    
"""
#prediction한 이미지 처리
image_paths = [
    "/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup_reduce4.5/5616*3744_inter4k_Raft_less_deg_x4_gloss_240807_large_patch_scratch_128_model_300/pexels-mccutcheon-1148998.png",
    "/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup_reduce4.5/5616*3744_basicvsr_pp_hf10_inter4k_Raft_less_deg_x4_gloss_240807_large_patch_scratch_128_model_300/pexels-mccutcheon-1148998.png",
    "/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup_reduce4.5/5616*3744_basicvsr_pp_decoderx10_inter4k_Raft_less_deg_x4_gloss_240807_large_patch_scratch_128/pexels-mccutcheon-1148998.png"
]

output_dir = "/mnt/sdb/VSR_Inference/4090x4_모래시계_testbackup_reduce4.5/lfhf"  # 결과 이미지를 저장할 디렉토리 경로
os.makedirs(output_dir, exist_ok=True)

image_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]  # 이미지를 흑백으로 불러옴

for idx, image in enumerate(image_list):
    score, edges = sobel_edge_score(image)
    print(f"Image {idx+1} - Sobel edge score: {score}")
    
    #각각의 image에 lf_mask, hf_mask 씌운후, edge score계산 
    lf_each_image_score, lf_edges = sobel_edge_score(image * lf_mask)
    hf_each_image_score, hf_edges = sobel_edge_score(image * hf_mask)


    print(f"Image {idx+1} - LF score: {lf_each_image_score}, HF score: {hf_each_image_score}")

    # LF/HF 이미지를 파일로 저장
    lf_output_path = os.path.join(output_dir, f"image_{idx+1}_lf.png")
    hf_output_path = os.path.join(output_dir, f"image_{idx+1}_hf.png")
    
    cv2.imwrite(lf_output_path, lf_edges) #lf_image)
    cv2.imwrite(hf_output_path, hf_edges) #hf_image)
    
    print(f"Saved LF image to: {lf_output_path}")
    print(f"Saved HF image to: {hf_output_path}")
"""
