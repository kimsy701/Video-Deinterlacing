import torch
from model.unimatch import UniMatch
from torchvision.transforms import functional as TF

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

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
    
    flow_extractor = UniMatch(feature_channels=128,
                              num_scales=2,
                              upsample_factor=8//2,
                              num_head=1,
                              ffn_dim_expansion=4,
                              num_transformer_layers=6,
                              reg_refine=True,
                              task='flow')
    fe_sd = torch.load('./pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')['model']
    print(flow_extractor.load_state_dict(fe_sd))
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

        with torch.no_grad():
            flow10 = flow_extractor(img_pad_list[1], img_pad_list[0],
                                  attn_type='swin',
                                  attn_splits_list=[2,8],
                                  corr_radius_list=[-1,4],
                                  prop_radius_list=[-1,1],
                                  num_reg_refine=6,
                                  first_scaling=args.first_scaling,
                                  task='flow')['flow_preds'][-1][:,:,:Huni,:Wuni] # [B, 2, H, W]
            flowt12 = flow_extractor(img_pad_list[1], img_pad_list[2],
                                  attn_type='swin',
                                  attn_splits_list=[2,8],
                                  corr_radius_list=[-1,4],
                                  prop_radius_list=[-1,1],
                                  num_reg_refine=6,
                                  first_scaling=args.first_scaling,
                                  task='flow')['flow_preds'][-1][:,:,:Huni,:Wuni] # [B, 2, H, W]
            flow01 = flow_extractor(img_pad_list[0], img_pad_list[1],
                                    attn_type='swin',
                                    attn_splits_list=[2,8],
                                    corr_radius_list=[-1,4],
                                    prop_radius_list=[-1,1],
                                    num_reg_refine=6,
                                    first_scaling=args.first_scaling,
                                    task='flow')['flow_preds'][-1][:,:,:Huni,:Wuni] # [B, 2, H, W]
            flowt21 = flow_extractor(img_pad_list[2], img_pad_list[1],
                                  attn_type='swin',
                                  attn_splits_list=[2,8],
                                  corr_radius_list=[-1,4],
                                  prop_radius_list=[-1,1],
                                  num_reg_refine=6,
                                  first_scaling=args.first_scaling,
                                  task='flow')['flow_preds'][-1][:,:,:Huni,:Wuni] # [B, 2, H, W]

            flow10_np = flow10[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            flow12_np = flowt12[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            flow01_np = flow01[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            flow21_np = flowt21[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            
            # print(np.min(flowt0_np), np.max(flowt0_np), np.mean(flowt0_np)) 
            #-2.744 2.9 0.2133, -26.75 48.22 2.156, -163.2 83.5 -2.502, -241.4 194.4 4.52

            np.save(os.path.join(newfolder, 'flow10.npy'), flow10_np)
            np.save(os.path.join(newfolder, 'flow12.npy'), flow12_np)
            np.save(os.path.join(newfolder, 'flow01.npy'), flow01_np)
            np.save(os.path.join(newfolder, 'flow21.npy'), flow21_np)
