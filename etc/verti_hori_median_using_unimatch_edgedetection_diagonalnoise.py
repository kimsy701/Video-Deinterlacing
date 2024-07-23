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
from torchvision.transforms import functional as TF
import cv2
from flow_visualization import flow_to_image

device = 0
torch.cuda.set_device(device)

def extract_number(directory):
    parts = directory.split('/')[-1]
    parts_fi = parts.split('_')
    if len(parts_fi) > 1:
        second_part = parts_fi[1][3:] #original folder number
        third_part =parts_fi[2][-1] #subfolder number
        try:
            return int(second_part)*1000 +int(third_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)S
    return float('inf')  # Return infinity if no underscore is found (optional)

def extract_number2(directory):
    parts = directory.split('r')[-1]
    return int(parts)

def extract_number3(directory):
    parts = directory.split('/')[-1]
    parts_fi = os.path.splitext(parts.split('_')[-1])[0]
    return int(parts_fi)

def extract_number4(directory):
    parts = directory.split('/')[-1]
    parts_fi = os.path.splitext(parts)[0]
    return int(parts_fi)

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

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    
    UNKNOWN_FLOW_THRESH = 1e7

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def save_flow_image(flow, output_path):
    flow_img = flow_to_image(flow)
    Image.fromarray(flow_img).save(output_path)
    
def median_filter_custom(data, size=3):
    padded_data = np.pad(data, pad_width=size//2, mode='edge')
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        window = padded_data[i:i+size]
        filtered_data[i] = np.median(window)
    return filtered_data
    
def save_forward_backward_np_flow(forward_flow, backward_flow, save_path, for_filename, back_filename): 
    np.save(os.path.join(save_path, for_filename), forward_flow)
    np.save(os.path.join(save_path, back_filename), backward_flow)
    print("saved flow numpy files")
    
def save_optical_flow_image(flow, output_path):
    """
    Save optical flow as a color-coded image.
    :param flow: Optical flow map of shape [H, W, 2]
    :param output_path: Path to save the color-coded image
    """
    flow_img = flow_to_image(flow)
    Image.fromarray(flow_img).save(output_path)
    
    
    
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
        self.flow_extractor = self.flow_extractor.to(device)
        fe_sd = torch.load('/home/nick/vsr/utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')['model']
        print(f'Optical Flow Estimator for Pseudo GT', self.flow_extractor.load_state_dict(fe_sd))
        for n,p in self.flow_extractor.named_parameters():
            p.requires_grad = False
        self.gt_root = gt_root
        self.folder_list = sorted(glob.glob(os.path.join(self.gt_root, '*')))
        
    
    def forward(self, im, im_after, im_before, img_index, threshold, vertical_size,flow_save_path, im_idx):
        print("im.shape",im.shape)
        flow = []
        Hori,Wori, C = im.shape
        
        #(720x486) -> (720x480)
        # im=im[3:-3,:,:] #처음에 해줌 
        im_after=im_after[3:-3,:,:]
        im_before=im_before[3:-3,:,:]
        
        unimatch_multiple = 128
        im = (torch.from_numpy(im)).permute(2,0,1).unsqueeze(0)
        im_after = (torch.from_numpy(im_after)).permute(2,0,1).unsqueeze(0)
        im_before = (torch.from_numpy(im_before)).permute(2,0,1).unsqueeze(0)
        
        img_list = [im_before,im,im_after]

        _,_,Huni,Wuni = img_list[0].size()
        padw = unimatch_multiple - (Wuni%unimatch_multiple) if Wuni%unimatch_multiple!=0 else 0
        padh = unimatch_multiple - (Huni%unimatch_multiple) if Huni%unimatch_multiple!=0 else 0
        img_pad_list = [TF.pad(img, (0,0,padw,padh), padding_mode='symmetric').to(device) for img in img_list]


        forward_temp = self.flow_extractor(img_pad_list[1],img_pad_list[2],  #im, im_after,
                                attn_type='swin',
                                attn_splits_list=[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
                                first_scaling=1,
                                task='flow')['flow_preds'][-1][:,:,:Huni,:Wuni] # [B, 2, H, W]
        flow.append(forward_temp)
            # flow10
        backward_temp = self.flow_extractor(img_pad_list[0],img_pad_list[1],       #im, im_before,
                                attn_type='swin',
                                attn_splits_list=[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
                                first_scaling=1,
                                task='flow')['flow_preds'][-1][:,:,:Huni,:Wuni]  # [B, 2, H, W]
        
        #save flow to numpy 
        forward_temp_np = forward_temp[0].permute(1,2,0).cpu().numpy().astype(np.float16)
        backward_temp_np = backward_temp[0].permute(1,2,0).cpu().numpy().astype(np.float16)
        for_file_name=f'for_{img_index}.npy'
        back_file_name=f'back_{img_index}.npy'

        save_forward_backward_np_flow(forward_temp_np, backward_temp_np, flow_save_path, for_file_name, back_file_name)
        
        
        #save flow to image 
        flow.append(backward_temp)
        flow = torch.stack(flow, axis=0) 
        # print("flow shape in unimatch", flow.shape) #torch.Size([2, 1, 2, 480, 720])
        
        # Compute square root of the sum of squares of x and y components
        sum_of_squares = torch.sqrt(torch.sum(flow ** 2, dim=2, keepdim=True))  # [2, 1, 1, 480, 720]
        mean_flow = torch.mean(sum_of_squares, axis=0)  #mean_flow shape torch.Size([1, 1, 480, 720])
        
        flow_path=f"/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_vertihori_median_th{threshold}_size{vertical_size}"
        os.makedirs(flow_path, exist_ok=True)
        # Visualize flow
        forward_flow_image_path = f"{flow_path}/{im_idx}_for_{img_index}_visualized.png"
        backward_flow_image_path = f"{flow_path}/{im_idx}_back_{img_index}_visualized.png"
        
        save_optical_flow_image(forward_temp_np, forward_flow_image_path)
        save_optical_flow_image(backward_temp_np, backward_flow_image_path)
        
        
        return mean_flow

"""
class EdgeDetection(nn.Module):
    def __init__(self, threshold1, threshold2):
        super(EdgeDetection, self).__init__()

        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, img):

        # 이미지 읽기
        if img is None:
            raise ValueError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
        
        sigma=0.33
        
        # compute the median of the single channel pixel intensities
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = np.median(img_gray)
        print("v",v) #0.45, 
        
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v)) #0.30
        upper = int(min(255, (1.0 + sigma) * v)) #0.59
        
        gray_image_8bit = cv2.convertScaleAbs(img_gray)
        # 엣지 감지
        edges = cv2.Canny(gray_image_8bit, lower, upper)
        
        # 엣지 부분의 주변 픽셀을 1로 설정하기 위한 커널
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 엣지 부분의 주변 영역을 1로 설정
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # 연결되지 않은 부분을 마스킹
        mask = np.ones_like(edges, dtype=np.uint8)
        mask[dilated_edges > 0] = 0  # 엣지가 있는 부분은 0, 없는 부분은 1

        return mask
"""
    
class EdgeDetection(nn.Module):
    def __init__(self):
        super(EdgeDetection, self).__init__()

    def forward(self, img):
        """
        엣지 감지 적용 및 마스킹
        :param img: 입력 이미지
        :return: 낮은 임계값과 높은 임계값의 엣지 차이를 마스킹한 이미지
        """
        if img is None:
            raise ValueError("이미지를 읽을 수 없습니다. 이미지를 확인하세요.")

        # 그레이스케일로 변환
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sigma=0.33
        
        # compute the median of the single channel pixel intensities
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) *255
        
        v = np.median(img_gray)
        # v = np.median(img)
        print("v",v) #0.45, 
        
        # apply automatic Canny edge detection using the computed median
        lower = max(0, (1.0 - sigma) * v) #0.30
        upper = min(255, (1.0 + sigma) * v)#0.59
        
        print("lower",lower,"upper",upper,1)
        
        gray_image_8bit = cv2.convertScaleAbs(img_gray)
        # gray_image_8bit = cv2.convertScaleAbs(img)

        # 낮은 임계값으로 엣지 감지
        # edges_low = cv2.Canny(gray_image_8bit, lower, upper/5)
        edges_low = cv2.Canny(gray_image_8bit, lower, upper)

        # 높은 임계값으로 엣지 감지
        edges_high = cv2.Canny(gray_image_8bit, lower, upper * 1.5)

        # 두 엣지 간의 차이를 계산
        edge_diff = cv2.absdiff(edges_low, edges_high)

        # 엣지 부분의 주변 픽셀을 1로 설정하기 위한 커널
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 엣지 부분의 주변 영역을 1로 설정
        dilated_edges = cv2.dilate(edge_diff, kernel, iterations=1)

        # 차이 부분을 마스킹
        mask = np.ones_like(edge_diff, dtype=np.uint8)
        mask[dilated_edges > 0] = 0  # 엣지 차이가 있는 부분은 0, 없는 부분은 1
        
        

        
        # 엣지 부분의 주변 픽셀을 1로 설정하기 위한 커널
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 엣지 부분의 주변 영역을 1로 설정
        dilated_edges1 = cv2.dilate(edges_low, kernel, iterations=1)
        dilated_edges2 = cv2.dilate(edges_high, kernel, iterations=1)

        # 연결되지 않은 부분을 마스킹
        mask1 = np.ones_like(edges_low, dtype=np.uint8)
        mask1[dilated_edges1> 0] = 0  # 엣지가 있는 부분은 0, 없는 부분은 1
        mask2 = np.ones_like(edges_high, dtype=np.uint8)
        mask2[dilated_edges2> 0] = 0  # 엣지가 있는 부분은 0, 없는 부분은 1


        return mask,lower,upper, mask1, mask2
    
    
class DiagonalNoise(nn.Module):
    def __init__(self, diff_thres=254, period=4):
        super(DiagonalNoise, self).__init__()
        self.diff_thres = diff_thres
        self.period = period

    def forward(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Define filters to detect diagonal patterns
        kernels = {
            # 'Diagonal TL-BR (small)': np.array([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]),  # Diagonal from top-left to bottom-right
            # 'Diagonal BL-TR (small)': np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]),  # Diagonal from bottom-left to top-right
            'Diagonal TL-BR (medium)': np.array([[-1, 1, 0, 0], [-1, 1, 0, 0], [-1, 1, 0, 0], [-1, 1, 0, 0]]),  # Medium diagonal from top-left to bottom-right
            'Diagonal BL-TR (medium)': np.array([[0, 1, -1, 0], [0, 1, -1, 0], [0, 1, -1, 0], [0, 1, -1, 0]])  # Medium diagonal from bottom-left to top-right
        }

        # Apply the filters to the image
        filtered_images = {name: cv2.filter2D(image, -1, kernel) for name, kernel in kernels.items()}

        # Post-processing to highlight detected diagonals
        # Convert to binary image where detected diagonal areas are white
        binary_images = {name: np.where(img > 0, 255, 0).astype(np.uint8) for name, img in filtered_images.items()}

        # Combine binary images to show all detected diagonals
        added_mask = np.array(list(binary_images.values()))[0] + np.array(list(binary_images.values()))[1]
        single_channel_mask = np.max(added_mask, axis=-1)
        # print("mask",single_channel_mask.shape)  #(480, 720)
        
        # print(np.array(mask))

        return single_channel_mask

    

    

# vertical median where flow is high(mask=1)
class VertiHoriMedian(nn.Module):
    def __init__(self):
        super(VertiHoriMedian, self).__init__()

    def forward(self, img, mask, vertical_size):
        medianed_image = img.copy()
        medianed_image=medianed_image[3:-3,:,:]
        print("img shape", img.shape)
        for col in range(img.shape[1]):
            for ch in range(img.shape[2]):
                col_data = img[:, col, ch]
                col_mask = mask[:, col]
                col_data = col_data[3:-3]
                col_data_filtered = col_data.copy()
                col_data_filtered = median_filter_custom(col_data, size=vertical_size)
                shifted_data = np.pad(col_data_filtered, (1, 0), mode='constant')[:-1]
                col_data_filtered_twice = median_filter_custom(shifted_data, size=3)
                medianed_image[:, col, ch] = col_data_filtered_twice

        for row in range(medianed_image.shape[0]):
            for ch in range(medianed_image.shape[2]):
                row_data = medianed_image[row, :, ch]
                row_data_filtered = median_filter_custom(row_data, size=vertical_size)
                shifted_data = np.pad(row_data_filtered, (1, 0), mode='constant')[:-1]
                row_data_filtered_twice = median_filter_custom(shifted_data, size=3)
                medianed_image[row, :, ch] = row_data_filtered_twice
                
        return medianed_image
    



########## process ###########
# Process images in the folder
def process_images_in_folder(folder_path,save_path,flow_save_path,flow_mask_save_path, edge_save_path,diag_save_path,fi_mask_save_path, unimatch, edgedetection, diagonalnoise, vertihorimedian, threshold, vertical_size):
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')), key=extract_number4)
    
    for i, image_path in enumerate(image_paths):
        print("i",i)
        img = load_image(image_path)
        img=img[3:-3,:,:]
        
        ##### flow #####
        if i == 0: #first image : only backward flow
            img_after = load_image(image_paths[i + 1])
            flow = unimatch(img, img_after, img_after, i, threshold, vertical_size,flow_save_path,i)
        elif i == len(image_paths) - 1:  #last image : only forward flow
            img_before = load_image(image_paths[i - 1])
            flow = unimatch(img, img_before, img_before,i, threshold, vertical_size,flow_save_path,i)
        else:
            time1=time.time()
            img_before = load_image(image_paths[i - 1])
            img_after = load_image(image_paths[i + 1])
            flow = unimatch(img, img_after, img_before,i, threshold, vertical_size,flow_save_path,i)
            time2=time.time()
        # print("flow.shape",flow.shape)  #torch.Size([1, 1, 480, 720])
        flow_mask = (flow > threshold).cpu().numpy().astype(np.uint8) #이미지 1개에 대한 flow mask 
        
        flow_mask_save_path = os.path.join(flow_mask_save_path, f'{i+1}.png')
        cv2.imwrite(flow_mask_save_path, flow_mask.squeeze(0).squeeze(0) * 255)  # Save as image
        
        ##### edge #####
        edge_mask, lower, upper, mask1, mask2=edgedetection(img)
        print("lower",lower,"upper",upper)
        edge_save_foler_path=f'{edge_save_path}_th1_{lower}_th2_{upper}'
        edge_save_foler_path1=f'{edge_save_path}_th1_{lower}_th2_{upper}_mask1'
        edge_save_foler_path2=f'{edge_save_path}_th1_{lower}_th2_{upper}_mask2'
        os.makedirs(edge_save_foler_path, exist_ok=True)
        os.makedirs(edge_save_foler_path1, exist_ok=True)
        os.makedirs(edge_save_foler_path2, exist_ok=True)
        edge_save_path = os.path.join(edge_save_foler_path, f'{i+1}.png')
        edge_save_path1 = os.path.join(edge_save_foler_path1, f'{i+1}.png')
        edge_save_path2 = os.path.join(edge_save_foler_path2, f'{i+1}.png')
        cv2.imwrite(edge_save_path, edge_mask * 255)  # Save as image
        cv2.imwrite(edge_save_path1, mask1 * 255)  # Save as image
        cv2.imwrite(edge_save_path2, mask2 * 255)  # Save as image
        
        ##### 대각선 노이즈 #####
        diagonal_mask=diagonalnoise(img)
        diagonal_save_path = os.path.join(diag_save_path, f'{i+1}.png')
        cv2.imwrite(diagonal_save_path, diagonal_mask*255)  # Save as image
        
        
        ##### final mask #####
        # flow_mask와 edge_mask의 AND 조건을 먼저 계산
        and_mask = np.logical_and(flow_mask.squeeze(0).squeeze(0) == 1, edge_mask == 1)

        # AND 조건 결과와 diagonal_mask의 OR 조건을 계산
        fi_mask = np.logical_or(and_mask, diagonal_mask == 1).astype(np.uint8)
        fi_mask_save_path = os.path.join(fi_mask_save_path, f'{i+1}.png')
        cv2.imwrite(fi_mask_save_path, fi_mask * 255)  # Save as image
        
        medianed_image = vertihorimedian(img, fi_mask, vertical_size)
        save_img_path=os.path.join(save_path,f'{i+1}.png')        
        save_image(medianed_image, save_img_path)
        

# Main script
if __name__ == '__main__':

    flow_threshold = 1 #1.0  # Change this part if needed
    vertical_size = 3 #얼마나 median 취할지 radius
    

    folder_path = '/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_twotime_sample'  # Update this path
    save_path = f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_twotime_sample_vertihorimedian_size{vertical_size}_flowth{flow_threshold}_edge_diagonal'
    os.makedirs(save_path, exist_ok=True)
    
    flow_save_path=f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_flow_{flow_threshold}'
    flow_mask_save_path=f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_flow_mask{flow_threshold}'
    edge_save_path=f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_edge'
    diag_save_path=f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_diag'
    fi_mask_save_path=f'/mnt/sde/deinter_datasets/GCP_backup/deinter_dataset/qtgmc_winter_rst_fi_mask_vertihorimedian'
    os.makedirs(flow_save_path, exist_ok=True)
    os.makedirs(flow_mask_save_path, exist_ok=True)
    os.makedirs(edge_save_path, exist_ok=True)
    os.makedirs(diag_save_path, exist_ok=True)
    os.makedirs(fi_mask_save_path, exist_ok=True)
    
    
    unimatch = Unimatch(gt_root=folder_path)
    edgedetection=EdgeDetection()
    diagonalnoise=DiagonalNoise()
    vertihorimedian = VertiHoriMedian()
    
    process_images_in_folder(folder_path,save_path,flow_save_path,flow_mask_save_path, edge_save_path,diag_save_path,fi_mask_save_path,unimatch, edgedetection, diagonalnoise, vertihorimedian, flow_threshold,vertical_size)
