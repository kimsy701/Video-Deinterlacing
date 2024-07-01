import sys
sys.path.append('core')
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from tqdm import tqdm
from utils.utils import resize_data, load_ckpt



def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=8, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        epe_list = np.array([], dtype=np.float32)
        px1_list = np.array([], dtype=np.float32)
        px3_list = np.array([], dtype=np.float32)
        px5_list = np.array([], dtype=np.float32)
        print(f"load data success {len(val_loader)}")
        for i_batch, data_blob in enumerate(val_loader):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            flow, info = calc_flow(args, model, image1, image2)
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
            px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
            px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
            epe = epe.mean(dim=[1, 2]).cpu().numpy()
            epe_list = np.append(epe_list, epe)
            px1_list = np.append(px1_list, px1)
            px3_list = np.append(px3_list, px3)
            px5_list = np.append(px5_list, px5)

        epe = np.mean(epe_list)
        px1 = np.mean(px1_list)
        px3 = np.mean(px3_list)
        px5 = np.mean(px5_list)
        print(f"Validation {dstype} EPE: {epe}, 1px: {100 * (1 - px1)}")


@torch.no_grad()
def validate_kitti(args, model, input_h, input_w):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = datasets.KITTI(split='training')
    # val_dataset = datasets.KITTI(split='testing')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    print(f"load data success {len(val_loader)}")
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    
    total_flow=torch.zeros(len(val_loader), 2, input_h, input_w) #x,y방향으로 2는 fix
    total_flow_2=torch.zeros(len(val_loader), 2, input_h//2, input_w//2) #x,y방향으로 2는 fix
    total_flow_4=torch.zeros(len(val_loader), 2, input_h//4, input_w//4) #x,y방향으로 2는 fix
    total_flow_8=torch.zeros(len(val_loader), 2, input_h//8, input_w//8) #x,y방향으로 2는 fix
    
    forward_flows=[]
    backward_flows=[]
    
    for i_batch, data_blob in enumerate(val_loader):
        print("i_batch",i_batch)
        for_sc4_flows=[]
        back_sc4_flows=[]
        
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        # image1, image2 = [x.cuda(non_blocking=True) for x in data_blob]
        for_flow, for_info = calc_flow(args, model, image1, image2)
        back_flow, back_info = calc_flow(args, model, image2, image1)
        # print("flow shape", flow.shape) #torch.Size([1, 2, 375, 1242]) #두장 계산해서 나온 값 
     
        new_size1 = (input_h // 2, input_w // 2)
        new_size2 = (input_h // 4, input_w // 4)
        new_size3 = (input_h // 8, input_w // 8)
     
        ### flow for 4 scales 방법 1 ###
        
        # image1_2 =  F.interpolate(image1, size=new_size1, mode='bicubic', align_corners=False)#torch.Size([1, 2, 375//2, 1242//2]) 
        # image2_2 = F.interpolate(image2, size=new_size1, mode='bicubic', align_corners=False)#torch.Size([1, 2, 375//2, 1242//2]) 
        
        # image1_4 =  F.interpolate(image1, size=new_size2, mode='bicubic', align_corners=False)#torch.Size([1, 2, 375//4, 1242//4]) 
        # image2_4 =  F.interpolate(image2, size=new_size2, mode='bicubic', align_corners=False)#torch.Size([1, 2, 375//4, 1242//4]) 
        
        # image1_8 = F.interpolate(image1, size=new_size3, mode='bicubic', align_corners=False)#torch.Size([1, 2, 375//8, 1242//8]) 
        # image2_8 = F.interpolate(image2, size=new_size3, mode='bicubic', align_corners=False)#torch.Size([1, 2, 375//8, 1242//8]) 
        
        # flow_2 = calc_flow(args, model, image1_2, image2_2)
        # flow_4 = calc_flow(args, model, image1_4, image2_4)
        # flow_8 = calc_flow(args, model, image1_8, image2_8)
        
        ### flow for 4 scales 방법 2 ###
        
        for_flow_2 = F.interpolate(for_flow, size = new_size1, mode='bicubic', align_corners=False)
        for_flow_4 = F.interpolate(for_flow, size = new_size2, mode='bicubic', align_corners=False)
        for_flow_8 = F.interpolate(for_flow, size = new_size3, mode='bicubic', align_corners=False)
        
        back_flow_2 = F.interpolate(back_flow, size = new_size1, mode='bicubic', align_corners=False)
        back_flow_4 = F.interpolate(back_flow, size = new_size2, mode='bicubic', align_corners=False)
        back_flow_8 = F.interpolate(back_flow, size = new_size3, mode='bicubic', align_corners=False)
        
        
                
    #     epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    #     mag = torch.sum(flow_gt**2, dim=1).sqrt()
    #     val = valid_gt >= 0.5
    #     out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    #     for b in range(out.shape[0]):
    #         epe_list = np.append(epe_list, epe[b][val[b]].mean().cpu().numpy())
    #         out_valid_pixels += out[b][val[b]].sum().cpu().numpy()
    #         num_valid_pixels += val[b].sum().cpu().numpy()
    
    # epe = np.mean(epe_list)
    # f1 = 100 * out_valid_pixels / num_valid_pixels
    # print("Validation KITTI: %f, %f" % (epe, f1))
    # return {'kitti-epe': epe, 'kitti-f1': f1}
        for_sc4_flows.append(for_flow)
        for_sc4_flows.append(for_flow_2)
        for_sc4_flows.append(for_flow_4)
        for_sc4_flows.append(for_flow_8)
        
        forward_flows.append(for_sc4_flows)
        
        
        back_sc4_flows.append(back_flow)
        back_sc4_flows.append(back_flow_2)
        back_sc4_flows.append(back_flow_4)
        back_sc4_flows.append(back_flow_8)
        
        backward_flows.append(back_sc4_flows)

    return forward_flows, backward_flows

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = datasets.SpringFlowDataset(split='val')
    val_loader = data.DataLoader(val_dataset, batch_size=4, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    epe_list = np.array([], dtype=np.float32)
    px1_list = np.array([], dtype=np.float32)
    px3_list = np.array([], dtype=np.float32)
    px5_list = np.array([], dtype=np.float32)
    print(f"load data success {len(val_loader)}")
    pbar = tqdm(total=len(val_loader))
    
    for i_batch, data_blob in enumerate(val_loader):
        print("i_batch",i_batch)
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        
        
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
        px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
        px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
        epe = epe.mean(dim=[1, 2]).cpu().numpy()
        epe_list = np.append(epe_list, epe)
        px1_list = np.append(px1_list, px1)
        px3_list = np.append(px3_list, px3)
        px5_list = np.append(px5_list, px5)
        pbar.update(1)

    pbar.close()
    epe = np.mean(epe_list)
    px1 = np.mean(px1_list)
    px3 = np.mean(px3_list)
    px5 = np.mean(px5_list)

    print(f"Validation Spring EPE: {epe}, 1px: {100 * (1 - px1)}")

@torch.no_grad()
def validate_middlebury(args, model):
    """ Peform validation using the Middlebury (public) split """
    val_dataset = datasets.Middlebury()
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    print(f"load data success {len(val_loader)}")
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        val = valid_gt >= 0.5
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        for b in range(out.shape[0]):
            epe_list = np.append(epe_list, epe[b][val[b]].mean().cpu().numpy())
            out_valid_pixels += out[b][val[b]].sum().cpu().numpy()
            num_valid_pixels += val[b].sum().cpu().numpy()
    
    epe = np.mean(epe_list)
    f1 = 100 * out_valid_pixels / num_valid_pixels
    print("Validation middlebury: %f, %f" % (epe, f1))

def eval(args):
    # torch.cuda.set_device(args.gpus)
    args.gpus = args.gpus
    
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()
    input_h = args.input_h
    input_w = args.input_w    
    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, model)
        elif args.dataset == 'sintel':
            validate_sintel(args, model)
        elif args.dataset == 'kitti':
            forward_flows, backward_flows= validate_kitti(args, model, input_h, input_w)
            # print("len(total_flows)",len(total_flows)) #2
            # print("len(total_flows[0])",len(total_flows[0])) #4
            # print("len(total_flows[0][0])",len(total_flows[0][0])) #1
            # print("len(total_flows[0][0][0])",len(total_flows[0][0][0])) #2
            # print("len(total_flows[0][0][0][0])",len(total_flows[0][0][0][0])) #375 (가장 작은 h)
            # print("len(total_flows[0][0][0][0][0])",len(total_flows[0][0][0][0][0])) #1242 (가장 작은 w)
        elif args.dataset == 'middlebury':
            validate_middlebury(args, model)
    return forward_flows, backward_flows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    parser.add_argument('--input_h', help='input_height', required=True, type=int)
    parser.add_argument('--input_w', help='input_width', required=True, type=int)
    parser.add_argument('--gpus', help='GPU ID to use', required=True, type=list)
    args = parse_args(parser)
    forward_flows, backward_flows = eval(args)

if __name__ == '__main__':
    main()

  #  python get_flow.py --cfg config/eval/kitti-S.json --model models/Tartan-C-T-TSKH-kitti432x960-S.pth --input_h 375 --input_w 1242 --gpu [0]
