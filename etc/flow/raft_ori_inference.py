#### Pretrained RAFT 자체(학습 전)의 flow를 시각화 ####

import torch
# from model.unimatch import UniMatch
from torchvision.transforms import functional as TF

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

from raft import RAFT #raft.py에서 상대적 경로 부분 주석처리 해주고 절대적 경로로 바꿔주기!!



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Unimatch optical flow estimation.")
    parser.add_argument("--total_gpus", type=int, default=2, help="number of CUDA GPUs to use")
    parser.add_argument("--start_cuda_index", type=int, default=0, help="starting CUDA GPU index")
    parser.add_argument("--cuda_index", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--root", type=str, default='../datasets/256_new/sequences', help="root to load images")
    parser.add_argument("--save_root", type=str, default='../datasets/256_new_unimatch_flow/sequences', help="root to save flows")
    
    parser.add_argument("--first_scaling", type=int, default=1, help="downsizing ratio before computing flow")
    
    args = parser.parse_args()
    
    total_gpus = args.total_gpus
    device = args.cuda_index
    torch.cuda.set_device(device)
    root = args.root
    save_root = args.save_root
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = v  # 'module.' 접두사 제거
            else:
                new_state_dict[k] = v
        return new_state_dict

    
    flow_extractor = RAFT()
    fe_sd = torch.load('/home/nick/hy/inshorts_vsr/my_model/raft_src/models/raft-small.pth')
 
    fe_sd=remove_module_prefix(fe_sd)

    print(flow_extractor.load_state_dict(fe_sd))
 
    # flow_extractor.load_state_dict(torch.load(fe_sd), strict=False)
    print("loaded raft")
    for n,p in flow_extractor.named_parameters():
        p.requires_grad = False
    flow_extractor = flow_extractor.to(device)
   
    
    # folder_list = sorted(glob.glob(os.path.join(root, '*', '*')))
    folder_list = sorted(glob.glob(os.path.join(root, '*')))
    print("folder_list",folder_list)
    per_gpu = (len(folder_list)//total_gpus)
    per_gpu = per_gpu+1 if len(folder_list)%total_gpus else per_gpu
    start = per_gpu*(device-args.start_cuda_index)
    end = start+per_gpu if start+per_gpu <= len(folder_list) else len(folder_list)
    # folder_list = folder_list[start:end]
    
    unimatch_multiple = 128
    print('Extract UniMatch Optical Flow')
    for folder in tqdm(folder_list):
        newfolder = os.path.join(save_root, '/'.join(folder.split('/')[-2:]))
        os.makedirs(newfolder, exist_ok=True)
        
        p1 = os.path.join(newfolder, 'flowt0.npy')
        p2 = os.path.join(newfolder, 'flowt1.npy')
        if os.path.exists(p1) and os.path.exists(p2): continue
        file_list = sorted(glob.glob(os.path.join(folder, '*.png'))\
                          +glob.glob(os.path.join(folder, '*.jpg')))
        
        img_list = [(torch.from_numpy(cv2.imread(file)[:,:,[2,1,0]])/255).permute(2,0,1).unsqueeze(0) for file in file_list]

        _,_,Huni,Wuni = img_list[0].size()
        padw = unimatch_multiple - (Wuni%unimatch_multiple) if Wuni%unimatch_multiple!=0 else 0
        padh = unimatch_multiple - (Huni%unimatch_multiple) if Huni%unimatch_multiple!=0 else 0
        img_pad_list = [TF.pad(img, (0,0,padw,padh), padding_mode='symmetric').to(device) for img in img_list]
        # img_pad_list = [TF.pad(img.half(), (0,0,padw,padh), padding_mode='symmetric').to(device) for img in img_list]
        # flow_extractor.half()

        with torch.no_grad():

            flow10 = flow_extractor(img_pad_list[1], img_pad_list[0], iters=1)[0][:,:,:Huni,:Wuni] # [B, 2, H, W]
            flow12 = flow_extractor(img_pad_list[1], img_pad_list[2], iters=1)[0][:,:,:Huni,:Wuni] # [B, 2, H, W]
            flow01 = flow_extractor(img_pad_list[0], img_pad_list[1], iters=1)[0][:,:,:Huni,:Wuni] # [B, 2, H, W]
            flow21 = flow_extractor(img_pad_list[2], img_pad_list[1], iters=1)[0][:,:,:Huni,:Wuni] # [B, 2, H, W]

            flow10_np = flow10[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            flow12_np = flow12[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            flow01_np = flow01[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            flow21_np = flow21[0].permute(1,2,0).cpu().numpy().astype(np.float16)

            np.save(os.path.join(newfolder, 'flow10.npy'), flow10_np)
            np.save(os.path.join(newfolder, 'flow12.npy'), flow12_np)
            np.save(os.path.join(newfolder, 'flow01.npy'), flow01_np)
            np.save(os.path.join(newfolder, 'flow21.npy'), flow21_np)
