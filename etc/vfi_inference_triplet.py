# [START_COMMAND]
# python3 -m vfi_inference --cuda_index 0 \
# --use_video --root ../VFI_Inference/thistest/test_video.mp4 --save_root ../VFI_Inference/thistest/results --source_frame_ext png \
# --pretrain_path ./pretrained/upr_freq002.pth \
# --pyr_level 7 --nr_lvl_skipped 0 --splat_mode average --down_scale 1 \
# --make_video --fps 0 --new_video_name test_video_vfi.mp4

# python3 -m vfi_inference_triplet --cuda_index 0 \
# --root ../VFI_Inference/thistriplet_notarget --pretrain_path ./pretrained/upr_freq002.pth \
# --pyr_level 7 --nr_lvl_skipped 0 --splat_mode average --down_scale 1

# [FILE SYSTEM]
# args.root 폴더 아래에
# 하위 폴더 (깊이는 최대 10개) 아래에
# triplet 3개 이미지 (with GT) 또는 triplet 2개 이미지 (without GT)

# [START_COMMAND]
#python vfi_inference_triplet.py --root=/mnt/sdb/deinter/deinter_dataset/train_val_winter_two_time_fi_bff_vfi_input_re

import argparse

import os
import cv2
import glob
import torch
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from utils.metrics import calculate_batch_psnr, calculate_batch_ssim

from modules.components.upr_freq_v1_7 import upr_freq as upr_freq002
from modules.components.upr_basic import upr as upr_basic

from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).cuda().eval()
# depth_anything = DepthAnything.from_pretrained('v1.7.pth').to(DEVICE).eval()
transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

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


@torch.no_grad()
def pred_depth(image):
    image = image[:,:,[2,1,0]]/255
    imaged = transform({'image': image})['image']
    imaged = torch.from_numpy(imaged).unsqueeze(0).cuda()
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda()
    h, w = image.shape[-2:]
    
    depth = depth_anything(imaged)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth

def multiple_pad(image, multiple):
    _,_,H,W = image.size()
    pad1 = multiple-(H%multiple) if H%multiple!=0 else 0
    pad2 = multiple-(W%multiple) if W%multiple!=0 else 0
    return TF.pad(image, (0,0,pad2,pad1))

print('인퍼런스 시\n1. utils.pad.py replicate->constant로 변경하고\n2. components upr Model 최초인풋에서 normalization과 padding 위치 바꿨는지 확인할 것 (padding이 위에 있어야됨)')
def main():
    NOW = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    parser = argparse.ArgumentParser('Video Frame Interpolation Inference',add_help=True)
    parser.add_argument('--cuda_index', default=0, type=int, help='CUDA GPU index')
    
    parser.add_argument('--exist_gt', action='store_true',  help='whether ground-truth existing') #file 3개 존재시, 2개만 있는 폴더라면 사용 x
    parser.add_argument('--root', default='', type=str, help='root containing frames [./triplet_name]') #폴더들이 있는 상위 폴더
    parser.add_argument('--pretrain_path', default='v1.7.pth', type=str, help='path containing pretrained model')
    
    parser.add_argument('--pyr_level', default=5, type=int, help='UPR-Net pyramid level') #256:3 , full HD: 7, 해상도에 따라 달라짐 
    parser.add_argument('--nr_lvl_skipped', default=0, type=int, help='UPR-Net pyramid skip number') #건드리지 x
    parser.add_argument('--splat_mode', default='average', type=str, help='UPR-Net warping splat mode')#건드리지 x
    parser.add_argument('--down_scale', default=1, type=int, help='frame down-scaling factor (due to GPU memory issue)')#건드리지 x
    
    args = parser.parse_args()
    assert not args.root.endswith('/'), f"'root' ({args.root}) must not end with '/'"
    
    DEVICE = args.cuda_index
    torch.cuda.set_device(DEVICE)
    ROOT = args.root
    SAVE_ROOT = f'{ROOT}_{NOW}' #원래 root+날짜 
    os.makedirs(SAVE_ROOT, exist_ok=True)
    SCALE = args.down_scale
    
    model = upr_freq002.Model(pyr_level=args.pyr_level,
                              nr_lvl_skipped=args.nr_lvl_skipped,
                              splat_mode=args.splat_mode)
    sd = torch.load(args.pretrain_path, map_location='cpu')
    sd = sd['model'] if 'model' in sd.keys() else sd
    print(model.load_state_dict(sd))
    model = model.to(DEVICE)
    
    star = '/*'
    temp = [x for i in range(10) for x in glob.glob(f'{ROOT}{star*i}') if os.path.isfile(x)]
 
    folder_list = sorted(set([os.path.split(x)[0] for x in temp]), key=extract_number2)
    print("folder_list",folder_list[:10])
    if args.exist_gt:
        with open(os.path.join(SAVE_ROOT, f'record.txt'), 'w', encoding='utf8') as f:
            f.writelines('')
            psnr_list = []
            ssim_list = []
    
    print('@@@@@@@@@@@@@@@@@@@@Staring VFI@@@@@@@@@@@@@@@@@@@@')
    for folder in tqdm(folder_list):
        file_list = []
        for ext in ['tif', 'TIF', 'jpg', 'png', 'tga', 'TGA']:
            file_list += sorted(glob.glob(os.path.join(folder, f'*.{ext}')))
        cur_ext = os.path.splitext(file_list[0])[1][1:]
        if cur_ext in ['tga', 'TGA']:
            img_list = [TF.to_tensor(Image.open(file))[:3].unsqueeze(0).to(DEVICE) for file in file_list]
        else:
            img_list = [(torch.from_numpy(cv2.imread(file)[:,:,[2,1,0]])/255).permute(2,0,1).unsqueeze(0).to(DEVICE) for file in file_list]

        _,_,Hori,Wori = img_list[0].size()
#         if Hori*Wori<=2100000:
#             SCALE = 1
#         elif Hori*Wori<=2100000*4:
#             SCALE = 2
#         else:
#             SCALE = 4
        if args.exist_gt:
            img_list = [multiple_pad(img, SCALE) if k!=1 else img for k, img in enumerate(img_list)]
            img_list = [F.interpolate(img, scale_factor=1/SCALE, mode='bicubic') if k!=1 else img for k, img in enumerate(img_list)]
            img0,imgt,img1 = img_list
        else:
            img_list = [multiple_pad(img, SCALE) for k, img in enumerate(img_list)]
            img_list = [F.interpolate(img, scale_factor=1/SCALE, mode='bicubic') for k, img in enumerate(img_list)]
            img0,img1 = img_list
        
        dep0, dep1 = pred_depth(cv2.imread(file_list[0])), pred_depth(cv2.imread(file_list[-1]))
        
        with torch.no_grad():
            result_dict, extra_dict = model(img0, img1, pyr_level=args.pyr_level, nr_lvl_skipped=args.nr_lvl_skipped, time_step=0.5, dep0=dep0, dep1=dep1)
            out = F.interpolate(result_dict['imgt_pred'], scale_factor=SCALE, mode='bicubic')[:,:,:Hori,:Wori].clamp(0,1)
            
            if args.exist_gt:
                psnr, _ = calculate_batch_psnr(imgt, out)
                ssim, _ = calculate_batch_ssim(imgt, out)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        
        filepath, ext = os.path.splitext(file_list[1])
        newfilename = filepath.replace(ROOT, SAVE_ROOT)
        newfile = newfilename+'_pred'+ext if args.exist_gt else os.path.join(os.path.split(newfilename)[0], 'im_pred'+ext)
        newfolder = os.path.split(newfile)[0]
        os.makedirs(newfolder, exist_ok=True)
        if cur_ext in ['tga', 'TGA']:
            TF.to_pil_image(out[0].cpu()).save(newfile)
        else:
            cv2.imwrite(newfile, (out[0].cpu().permute(1,2,0)*255).numpy().astype(np.uint8)[:,:,[2,1,0]])
        
        if args.exist_gt:
            with open(os.path.join(SAVE_ROOT, f'record.txt'), 'a', encoding='utf8') as f:
                foldername = '/'.join(folder.split('/')[2:])
                f.writelines(f'{foldername:45}PSNR: {psnr:.4f} SSIM: {ssim:.4f}\n')
    if args.exist_gt:
        print(f'PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.6f}')
        
if __name__ == '__main__':
    main()
