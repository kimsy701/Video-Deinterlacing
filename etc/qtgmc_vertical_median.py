##### Apply Vertical Median where flow is high in the QTGMC output (QTGMC has combing noise problem) #####
##### Use unimatch to calculate flow #####
from utils.unimatch.unimatch import UniMatch
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import median_filter
import os
import glob
from torchvision import transforms
from PIL import Image


# use unimatch and mask where flow is high
class Unimatch(nn.Module):
    def __init__(self, ):
        super(Unimatch, self).__init__()
        self.flow_extractor = UniMatch(feature_channels=128,
                                       num_scales=2,
                                       upsample_factor=8//2,
                                       num_head=1,
                                       ffn_dim_expansion=4,
                                       num_transformer_layers=6,
                                       reg_refine=True,
                                       task='flow')
        fe_sd = torch.load('./utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')['model']
        print(f'Optical Flow Estimator for Pseudo GT', self.flow_extractor.load_state_dict(fe_sd))
        for n,p in self.flow_extractor.named_parameters():
            p.requires_grad = False
        
        self.folder_list = sorted(glob.glob(os.path.join(self.gt_root, '*')))
    
    def forward(self, im, im_after, im_before):
        flow = []
        Hori,Wori, C = im[0].shape
        forward_temp = self.flow_extractor(im, im_after,
                                attn_type='swin',
                                attn_splits_list=[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
                                first_scaling=1,
                                task='flow')['flow_preds'][-1][:,:,:Hori,:Wori] # [B, 2, H, W]
        flow.append(forward_temp)
            # flow10
        backward_temp = self.flow_extractor(im, im_before,
                                attn_type='swin',
                                attn_splits_list=[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
                                first_scaling=1,
                                task='flow')['flow_preds'][-1][:,:,:Hori,:Wori] # [B, 2, H, W]
        flow.append(backward_temp)
        flow = torch.stack(flow, axis=0) 
        
        mean_flow=torch.mean(torch.abs(flow)) #average of abs(forward flow) and abs(backward flow) 
        
        return mean_flow


# vertical median where flow is high(mask=1)
class VerticalMedian(nn.Module):
    def __init__(self, ):
        super(VerticalMedian, self).__init__()
        
           
    def forward(self, img, mask):
        # Copy the input image to avoid modifying the original image
        medianed_image = img.copy()
        
        # Apply the median filter column by column where the mask is 1
        for col in range(img.shape[1]):
            for ch in range(img.shape[2]):
                col_data = img[:, col, ch]
                col_mask = mask[:, col]
                col_data_filtered = col_data.copy()
                col_data_filtered[col_mask == 1] = median_filter(col_data, size=3)[col_mask == 1]
                medianed_image[:, col, ch] = col_data_filtered
        
        return medianed_image
    
# Load and preprocess images
def load_image(image_path):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    return transform(image).numpy().transpose(1, 2, 0)  # Convert to (H, W, C) format

# Save processed image
def save_image(image, save_path):
    transform = transforms.ToPILImage()
    image = image.transpose(2, 0, 1)  # Convert back to (C, H, W) format
    image = transform(torch.from_numpy(image))
    image.save(save_path)
    

##### process ######
# Process images in the folder
def process_images_in_folder(folder_path, unimatch, verticalmedian, threshold):
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    
    for i, image_path in enumerate(image_paths):
        img = load_image(image_path)
        
        if i == 0: #first image : only backward flow
            img_after = load_image(image_paths[i + 1])
            flow = unimatch(img, img_after, img_after)
        elif i == len(image_paths) - 1:  #last image : only forward flow
            img_before = load_image(image_paths[i - 1])
            flow = unimatch(img, img_before, img_before)
        else:
            img_before = load_image(image_paths[i - 1])
            img_after = load_image(image_paths[i + 1])
            flow = unimatch(img, img_after, img_before)
        
        mask = np.any(flow > threshold, axis=2).astype(np.uint8)
        medianed_image = verticalmedian(img, mask)
        
        save_image(medianed_image, image_path)

# Main script
if __name__ == '__main__':
    folder_path = '/path/to/your/folder'  # Update this path
    threshold = 1.0  # Change this part if needed

    unimatch = Unimatch(gt_root=folder_path)
    verticalmedian = VerticalMedian()

    process_images_in_folder(folder_path, unimatch, verticalmedian, threshold)
