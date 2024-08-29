import cv2
import numpy as np
import torch
import torch.nn as nn

def sobel_edge_score(image):

    # Check if image is loaded correctly
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Apply Sobel operator in the x direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate the sum of all gradient magnitudes
    edge_score = np.sum(sobel_magnitude)
    
    return edge_score

class LowFreqFilter(nn.Module):
    def __init__(self, channel):
        super(LowFreqFilter, self).__init__()
        conv= nn.Conv2d(channel, channel, kernel_size=3, padding=1,bias=False)
        conv.weight.requires_grad = False
        self.register_buffer('lowpass_filter', conv.weight) #학습하지도 않으며, state_dict에도 포함 x (기존 모델에서 finetune가능할 수 있게)

    def forward(self, x):
        x_lowfreq = nn.functional.conv2d(x, self.lowpass_filter, padding=1)
        x_highfreq = x - x_lowfreq
        # ic(x.shape,'lowfreqfilterclass') # torch.Size([3, 3, 256, 256]) -> torch.Size([3, 64, 128, 128]) -> torch.Size([3, 64, 64, 64])
        # ic(x_lowfreq.shape,'lowfreqfilterclass') # torch.Size([3, 3, 256, 256])-> torch.Size([3, 64, 128, 128]) -> torch.Size([3, 64, 64, 64])
        # ic(x_highfreq.shape,'lowfreqfilterclass') # torch.Size([3, 3, 256, 256]) -> torch.Size([3, 64, 128, 128]) -> torch.Size([3, 64, 64, 64])
        return x_lowfreq, x_highfreq
    
def find_HF_LF(image):
    lowfreqfilter=LowFreqFilter(channel=3)
    lf_img, hf_img = lowfreqfilter(image)
    return lf_img, hf_img

######################################    Applying part     ######################################
#### Apply Sobel edge score to 전체 image #### 
image_path = "path_to_your_image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

score = sobel_edge_score(image)
print(f"Sobel edge score: {score}")


#### Apply Sobel edge score HF/LF part ####
lf_image, hf_image = find_HF_LF(image)
lf_score = sobel_edge_score(lf_image)
hf_score = sobel_edge_score(hf_image)
print("lf score:", lf_score, "hf_score:", hf_score)


