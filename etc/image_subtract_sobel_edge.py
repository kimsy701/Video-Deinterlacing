import torch
import torch.nn.functional as F
import cv2
import numpy as np


# Convert image to tensor
def image_to_tensor(image):
    if len(image.shape) == 3:  # If image is RGB (H, W, C)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # Convert to (C, H, W)
    else:  # If image is grayscale (H, W)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add a channel dimension -> (1, H, W)
    return image_tensor

# Convert tensor to image
def tensor_to_image(tensor):
    if tensor.shape[0] == 1:  # If tensor is grayscale (1, H, W)
        image = tensor.squeeze().detach().numpy().astype(np.uint8)  # Convert to (H, W)
    else:  # If tensor is RGB (C, H, W)
        image = tensor.permute(1, 2, 0).detach().numpy().astype(np.uint8)  # Convert to (H, W, C)
    return image

# Horizontal Convolution Function
def hori_conv(image_tensor):
    
    if image_tensor.dim() == 3:  # [C, H, W] format
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension [B, C, H, W]

    # Define horizontal blur kernel (1x15 averaging)
    ksize = (15, 1)  # (Height, Width)
    kernel = np.ones(ksize, dtype=np.float32) / np.prod(ksize)
    kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1,1,15,1)
    
    # Expand kernel to match the number of input channels
    in_channels = image_tensor.shape[1]
    kernel = kernel.expand(in_channels, 1, -1, -1)  # Shape (C,1,15,1)

    # Apply horizontal convolution to each channel
    # Add padding to ensure output size matches input size
    padding = (ksize[1] // 2, 0)  # Horizontal padding
    conv_result = F.conv2d(image_tensor, kernel, padding=padding, groups=in_channels)
    image_ = conv_result.squeeze(0)
    # # Define horizontal blur kernel (1x3 averaging)
    # horizontal_kernel = torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1,1,1,3)
    
    # # Apply horizontal convolution to each channel separately
    # image_ = []
    # for c in range(image_tensor.shape[0]):  # Loop over channels (assuming image_tensor is [C, H, W])
    #     # Apply padding to each channel separately
    #     channel_padded = F.pad(image_tensor[c:c+1], (1, 1, 0, 0), mode='constant')  # Pad width: left=1, right=1
    #     # Convolve each channel separately and collect results
    #     conv_result = F.conv2d(channel_padded.unsqueeze(0), horizontal_kernel, padding=0)
    #     image_.append(conv_result.squeeze(0))  # Remove batch dimension
        
    # # Stack the processed channels back together
    # image_ = torch.cat(image_, dim=0)
    
    return image_

# Sobel Vertical Edge Detection Function
def sobel_verti_value(image_tensor):
    # Define the Sobel vertical kernel
    sobel_kernel_verti = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)
    
    # Apply convolution to each channel separately
    sobel_verti_imgs = []
    edge_intensity_values = []
    
    # Iterate over channels
    for c in range(image_tensor.shape[0]):  # Loop over channels (assuming image_tensor is [C, H, W])
        # Apply padding to each channel separately and convolve with the Sobel kernel
        channel_padded = F.pad(image_tensor[c:c+1], (1, 1, 1, 1), mode='constant')  # Pad width: left=1, right=1, top=1, bottom=1
        sobel_verti_img = F.conv2d(channel_padded.unsqueeze(0), sobel_kernel_verti, padding=0)
        
        # Calculate the edge intensity for each channel
        edge_intensity_value = torch.sum(torch.abs(sobel_verti_img)).item()
        
        sobel_verti_imgs.append(sobel_verti_img.squeeze(0))  # Remove batch dimension
        edge_intensity_values.append(edge_intensity_value)
    
    # Stack the processed channels back together
    sobel_verti_img = torch.cat(sobel_verti_imgs, dim=0)
    
    # Calculate the total number of pixels in the image
    num_pixels = image_tensor.size(1) * image_tensor.size(2)  # H * W
    
    # Normalize the edge intensity values by the number of pixels
    total_edge_intensity_value = sum(edge_intensity_values) / (image_tensor.shape[0] * num_pixels)
    
    return sobel_verti_img, total_edge_intensity_value

############################################  process  ############################################
#image 1 불러오기
#image 2 불러오기 
#subtracted_img = image1-image2

# Load image 1
image1 = cv2.imread('/mnt/sdb/VSR_Inference/순풍_noise/new_deint/00000488.tiff')  
# Load image 2
image2 = cv2.imread('/mnt/sdb/VSR_Inference/순풍_noise/noise/00000488.tiff') 

# Ensure the images are the same size
if image1.shape == image2.shape:
    # Subtract the images
    subtracted_img = cv2.subtract(image1, image2)
    
    # Save the result to a file
    cv2.imwrite('/mnt/sdb/VSR_Inference/순풍_noise/noise-new_deint/00000488.tiff', subtracted_img)  # Save with the desired file name
else:
    print("Error: The images have different dimensions and cannot be subtracted.")


#subtracted_img 의 sobel edge점수 구하기
# image = cv2.imread('/mnt/sdb/VSR_Inference/순풍_noise/noise/00000488.tiff', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('/mnt/sdb/VSR_Inference/순풍_noise/noise-new_deint/00000488.tiff')

# 2. Convert the image to a PyTorch tensor
image_tensor = image_to_tensor(image)

# 3. Apply horizontal convolution (blurring in the horizontal direction)
# horizontal_blurred_image_tensor = hori_conv_strong(image_tensor)
horizontal_blurred_image_tensor = hori_conv(image_tensor)

# 4. Calculate vertical edge using the Sobel kernel
vertical_edge_image, verti_edge_value = sobel_verti_value(horizontal_blurred_image_tensor)
print("verti_edge_value:",verti_edge_value)

# 5. Convert the tensors back to images
horizontal_blurred_image = tensor_to_image(horizontal_blurred_image_tensor)
vertical_edge_image = tensor_to_image(vertical_edge_image)

# Save the results using OpenCV
cv2.imwrite('/mnt/sdb/VSR_Inference/순풍_noise/noise-new_deint/00000488_hori_blur.png', horizontal_blurred_image)
cv2.imwrite('/mnt/sdb/VSR_Inference/순풍_noise/noise-new_deint/00000488_vertical_edge.png', vertical_edge_image)
