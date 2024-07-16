##### Apply Vertical Median where flow is high in the QTGMC output (QTGMC has combing noise problem) #####
##### code written at 4090x4-1_nick #####
##### Use unimatch to calculate flow #####
# from utils.unimatch.unimatch import UniMatch
from unimatch.unimatch import UniMatch
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import median_filter
import os
import glob
from torchvision import transforms
from PIL import Image
import time
import matplotlib.pyplot as plt

def visualize_mean_flow_with_values(mean_flow, save_path, filename):
    """
    Visualizes the mean optical flow with magnitude and direction, and displays the flow values in the image.
    Saves the visualization as a PNG image.
    
    Parameters:
    mean_flow (torch.Tensor): Mean flow tensor with shape [1, 2, 480, 720]
    save_path (str): The directory where the image will be saved
    filename (str): The name of the image file
    """
    # Squeeze the batch dimension
    mean_flow = mean_flow.squeeze(0)  # Now mean_flow has shape [2, 480, 720]
    
    # Separate the mean flow into x and y components
    flow_x = mean_flow[0].cpu().numpy()
    flow_y = mean_flow[1].cpu().numpy()
    
    # Compute flow magnitude and direction
    flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
    flow_direction = np.arctan2(flow_y, flow_x)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the flow magnitude as color-coded image using 'viridis' colormap
    im = ax.imshow(flow_magnitude, cmap='viridis', vmin=0, vmax=5)  # Adjust vmin and vmax as needed
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Flow Magnitude')
    
    # Plot the flow direction as quiver plot
    # ax.quiver(flow_x, flow_y, scale=1, scale_units='xy', angles='xy', color='white')
    
    # Plot flow values with arrows
    arrow_size = 1
    for y in range(0, flow_magnitude.shape[0], 10):
        for x in range(0, flow_magnitude.shape[1], 10):
            ax.arrow(x, y, flow_x[y, x], flow_y[y, x], color='white', head_width=arrow_size, head_length=arrow_size)
    
    # Set the title and axis labels
    ax.set_title('Mean Optical Flow')
    ax.set_xlabel('Flow X')
    ax.set_ylabel('Flow Y')
    
    # Save the figure
    save_img_path = os.path.join(save_path, f"{filename}.png")
    plt.savefig(save_img_path, bbox_inches='tight')
    plt.close(fig)
    
    
    
def save_forward_backward_np_flow(forward_flow, backward_flow, save_path, for_filename, back_filename): 
    np.save(os.path.join(save_path, for_filename), forward_flow)
    np.save(os.path.join(save_path, back_filename), backward_flow)
    print("saved flow numpy files")
    
    
    
# use unimatch and mask where flow is high
class Unimatch(nn.Module):
    def __init__(self, gt_root):
        super(Unimatch, self).__init__()
        self.flow_extractor = UniMatch(feature_channels=128,
                                       num_scales=2,
                                       upsample_factor=8//2,
                                       num_head=1,
                                       ffn_dim_expansion=4,
                                       num_transformer_layers=6,
                                       reg_refine=True,
                                       task='flow')
        fe_sd = torch.load('/home/nick/vsr/utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')['model']
        print(f'Optical Flow Estimator for Pseudo GT', self.flow_extractor.load_state_dict(fe_sd))
        for n,p in self.flow_extractor.named_parameters():
            p.requires_grad = False
        self.gt_root = gt_root
        self.folder_list = sorted(glob.glob(os.path.join(self.gt_root, '*')))
        
    
    def forward(self, im, im_after, im_before, img_index, magnitude, threshold, vertical_size):
        print("im.shape",im.shape)
        flow = []
        Hori,Wori, C = im.shape
        
        #(720x486) -> (720x480)
        im=im[3:-3,:,:]
        im_after=im_after[3:-3,:,:]
        im_before=im_before[3:-3,:,:]
        

        im = (torch.from_numpy(im)).permute(2,0,1).unsqueeze(0)
        im_after = (torch.from_numpy(im_after)).permute(2,0,1).unsqueeze(0)
        im_before = (torch.from_numpy(im_before)).permute(2,0,1).unsqueeze(0)
        forward_temp = self.flow_extractor(im, im_after,
                                attn_type='swin',
                                attn_splits_list=[2,4],#[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
                                first_scaling=1,
                                task='flow')['flow_preds'][-1][:,:,:Hori,:Wori] # [B, 2, H, W]
        flow.append(forward_temp)
            # flow10
        backward_temp = self.flow_extractor(im, im_before,
                                attn_type='swin',
                                attn_splits_list=[2,4],#[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
                                first_scaling=1,
                                task='flow')['flow_preds'][-1][:,:,:Hori,:Wori] # [B, 2, H, W]
        
        #save flow to numpy 
        forward_temp_np = forward_temp[0].permute(1,2,0).cpu().numpy().astype(np.float16)
        backward_temp_np = backward_temp[0].permute(1,2,0).cpu().numpy().astype(np.float16)
        flow_save_path='/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_flow'
        for_file_name=f'for_{img_index}.npy'
        print("for_file_name",for_file_name)
        back_file_name=f'back_{img_index}.npy'

        save_forward_backward_np_flow(forward_temp_np, backward_temp_np, flow_save_path, for_file_name, back_file_name)
        
        
        #save flow to image 
        flow.append(backward_temp)
        flow = torch.stack(flow, axis=0) 
        # print("flow shape in unimatch", flow.shape) #torch.Size([2, 1, 2, 480, 720])
        
        mean_flow=torch.mean(torch.abs(flow), axis=0) #[1,1,2,480,720] 형태로, average of abs(forward flow) and abs(backward flow) 
        print("mean_flow.shape", mean_flow.shape) #torch.Size([1, 2, 480, 720])
        flow_path=f"/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_vertical_median_th{threshold}_size{vertical_size}"
        os.makedirs(flow_path, exist_ok=True)
        visualize_mean_flow_with_values(mean_flow*magnitude, flow_path, f"{img_index+1}")
        
        return mean_flow


# vertical median where flow is high(mask=1)
class VerticalMedian(nn.Module):
    def __init__(self, ):
        super(VerticalMedian, self).__init__()
        
           
    def forward(self, img, mask,vertical_size):
        # Copy the input image to avoid modifying the original image
        medianed_image = img.copy()
        medianed_image=medianed_image[3:-3,:,:]
        
        print("img shape", img.shape) # (486, 720, 3)
        
        # Apply the median filter column by column where the mask is 1
        for col in range(img.shape[1]): #720
            for ch in range(img.shape[2]): #3
                col_data = img[:, col, ch]
                col_mask = mask[:, col]
                
                #(720x486) to (720x480) (mask is already processed in unimatch class)
                col_data = col_data[3:-3] #(480,)
                
                col_data_filtered = col_data.copy()
                # col_data_filtered[col_mask == 1] = median_filter(col_data, size=vertical_size)[col_mask == 1] #size=3
                col_data_filtered = median_filter(col_data, size=vertical_size) #size=3

                medianed_image[:, col, ch] = col_data_filtered
        
        return medianed_image
    
# Load and preprocess images
def load_image(image_path):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    return transform(image).numpy().transpose(1, 2, 0)  # Convert to (H, W, C) format

# Save processed image
def save_image(image, save_img_path):
    transform = transforms.ToPILImage()
    image = image.transpose(2, 0, 1)  # Convert back to (C, H, W) format
    image = transform(torch.from_numpy(image))
    image.save(save_img_path)
    

##### process ######
# Process images in the folder
def process_images_in_folder(folder_path,save_path, unimatch, verticalmedian, threshold, magnitude, vertical_size):
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    print("image_paths",image_paths)
    
    for i, image_path in enumerate(image_paths):
        if i>1080:
            print("i",i)
            img = load_image(image_path)

            if i == 0: #first image : only backward flow
                img_after = load_image(image_paths[i + 1])
                flow = unimatch(img, img_after, img_after, i,magnitude, threshold, vertical_size)
            elif i == len(image_paths) - 1:  #last image : only forward flow
                img_before = load_image(image_paths[i - 1])
                flow = unimatch(img, img_before, img_before,i, magnitude, threshold, vertical_size)
            else:
                time1=time.time()
                img_before = load_image(image_paths[i - 1])
                img_after = load_image(image_paths[i + 1])
                flow = unimatch(img, img_after, img_before,i,magnitude, threshold, vertical_size)
                time2=time.time()
                print("unimatch time", time2-time1)
                
            # print("flow.shape",flow.shape) #torch.Size([1, 2, 480, 720])
            
            # Remove batch dimension and compute the flow magnitude
            flow = flow.squeeze(0)  # Now flow has shape [2, 480, 720]
            flow_magnitude = torch.mean(flow, dim=0)  # Compute magnitude, resulting in shape [480, 720]
            
            mask = (flow_magnitude > threshold).cpu().numpy().astype(np.uint8)
            time3=time.time()
            medianed_image = verticalmedian(img, mask, vertical_size)
            time4=time.time()
            print("vertical median time", time4-time3)
            save_img_path=os.path.join(save_path,f'{i+1}.png')
            
            save_image(medianed_image, save_img_path)

# Main script
if __name__ == '__main__':

    threshold = 0.5 #1.0  # Change this part if needed
    magnitude = 3 # how much to multiply flows, when visualizing flow
    vertical_size = 3 #얼마나 median 취할지 radius


    folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst'  # Update this path
    save_path = f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_vertical_median_full_image_size{vertical_size}'
    os.makedirs(save_path, exist_ok=True)
    
    unimatch = Unimatch(gt_root=folder_path)
    verticalmedian = VerticalMedian()
    
    process_images_in_folder(folder_path, save_path, unimatch, verticalmedian, threshold, magnitude,vertical_size)
