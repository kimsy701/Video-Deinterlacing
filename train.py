import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from tqdm import tqdm
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim=StructuralSimilarityIndexMeasure(data_range=1.0)    
from Dataset.CustomDataset import CustomDataset
import Models.DfConv_EkSA_deinter as DfRes_arch_nores_nlsa

import wandb
wandb.login()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="deinter_project_re",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-6,
    "architecture": "deinterlace module + spynet",
    "dataset": "vimeo 90k",
    "epochs": 74,
    }
)

###### set rank ####
rank=int(os.environ['LOCAL_RANK'])
# print("rank", rank)
torch.cuda.set_device(rank)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl',world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()
    
def save_optimizer_state(path, rank, step):
    if rank == 0:
        optimizer_ckpt = {
                    "step": step
                    }
        torch.save(optimizer_ckpt, "{}/optimizer-ckpt.pth".format(path))
        

def train(rank, world_size, args):
    setup_ddp(rank, world_size)


    #################
    # configurations
    #################
    device = torch.device('cuda', rank)
    data_mode = 'evenodd' 
    model_path = '/home/kimsy701/Video-Deinterlacing/Models/trained_modelscheckpoint_epoch_2.pth'
    N_in = 5
    predeblur, HR_in = False, False
    back_RBs = 7
    model = DfRes_arch_nores_nlsa.DfConv_EkSA(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in, w_TSA=False)
   
    
    #### dataset and dataloader ###############################################################
    # train_dataset_folder = './Dataset/train_val_100'
    # GT_dataset_folder = './Dataset/gt_val_100_re'
    
    train_dataset_folder = '/home/kimsy701/Video-Deinterlacing/Dataset/sampled_train_dataset' # 6장씩..!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    GT_dataset_folder = '/home/kimsy701/Video-Deinterlacing/Dataset/sampled_gt_frames_re' #5장씩 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # train_dataset = CustomDataset(train_dataset_folder, GT_dataset_folder, data_mode = data_mode)  # Implement this dataset class as needed
    train_dataset = CustomDataset(train_dataset_folder, GT_dataset_folder)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    
    
    validation_dataset_folder = '/home/kimsy701/Video-Deinterlacing/Dataset/train_val_10' # 6장씩..!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    val_GT_dataset_folder = '/home/kimsy701/Video-Deinterlacing/Dataset/gt_val_10' #5장씩 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    validation_dataset = CustomDataset(validation_dataset_folder, val_GT_dataset_folder)  # Implement this dataset class as needed
    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    #### loss function and optimizer ##########################################################
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch = 1
    
    #### DDP setup ###########################################################################
    if args.RESUME == True:
        model.load_state_dict(torch.load(model_path), strict=False)
        info_dict = torch.load(args.ckpt_file)
        epoch = info_dict["step"]+1
        
    # model.train()
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # with open("parameters.txt", 'w') as f:
    #         for i, (name, param) in enumerate(model.named_parameters()):
    #             f.write(f"{i}_Parameter name: {name}\n")  #왜 searaft없지.... evaluation 용 SEAraft들고 와서
    
    #### training loop ########################################################################
    num_epochs = args.epochs
    # epoch=3
    
    # for epoch in range(num_epochs):
    while epoch < num_epochs +1:
        print("epoch",epoch)
        model.train()
        
        
        
        # epoch=3
        
        
        
        train_sampler.set_epoch(epoch)  # Shuffle the dataset
        running_loss = 0.0
        
        for batch_idx, data in (enumerate(tqdm(train_loader))):
            imgs_LQ, imgs_GT, img_path_l = data
            # print("imgs_LQ[0][2] avg", torch.mean(imgs_LQ[0][2])) #0.
            # print("imgs_LQ shape", imgs_LQ.shape) #torch.Size([8, 6, 3, 128, 448])
            # print("imgs_GT shape", imgs_GT.shape) #torch.Size([8, 5, 256, 448, 3])
            
            imgs_LQ = imgs_LQ.to(device)
            imgs_GT = imgs_GT.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs_LQ)
            # print("output shape", output.shape) #torch.Size([8, 3, 256, 448]) ? (batch, sequence, c,h,w) 이어야하는데,,, #혹시 가운데 프레임? #torch.Size([8, 3, 256, 448])
            # print("imgs_GT shape", imgs_GT.shape) #torch.Size([8, 5, 256, 448, 3])  #(x,x,x,x,448) 여야함
            
            #실험
            imgs_GT = imgs_GT.permute(0, 1, 4, 2, 3) # torch.Size([8, 5, 3, 256, 448]) #6은 뭐지..
            # print("imgs_GT[0]",imgs_GT.shape) #torch.Size([8, 3, 256, 448])
            
            loss = criterion(output, imgs_GT[:,2,:,:,:]) #imgs_GT중 가운데꺼.... 
            loss.backward()
            optimizer.step()
            
            #calculate training psnr and ssim
            psnr_list = torch.zeros(output.shape[0])
            ssim_list = torch.zeros(output.shape[0])
            
            for img_idx in range(output.shape[0]): #0~7(batch size)
                if torch.mean((output[img_idx] - imgs_GT[:,2,:,:,:][img_idx]) * (output[img_idx]- imgs_GT[:,2,:,:,:][img_idx]))>0:
                    img_psnr =  -10 * math.log10(torch.mean((output[img_idx] - imgs_GT[:,2,:,:,:][img_idx]) * (output[img_idx] - imgs_GT[:,2,:,:,:][img_idx])))
                    img_ssim = torch.mean(ssim(output[img_idx].unsqueeze(0).detach().cpu(),imgs_GT[:,2,:,:,:][img_idx].unsqueeze(0).detach().cpu()))
                else:
                    img_psnr = -10 * math.log10(0.00001)
                    img_ssim = 0.00001
                # print("img_path_l",img_path_l)
                # first_img_path = img_path_l[img_idx]
                # cv2.imwrite(f'/home/kimsy701/Video-Deinterlacing/pred_img_{img_idx}.png', (output[img_idx]*255).int().cpu().permute(1,2,0).numpy())
                # cv2.imwrite(f'/home/kimsy701/Video-Deinterlacing/gt_img_{img_idx}.png', (imgs_GT[:,2,:,:,:][img_idx]*255).int().cpu().permute(1,2,0).numpy())
                    
                psnr_list[img_idx]=img_psnr
                ssim_list[img_idx]=img_ssim
                    
            running_loss += loss.item()
            
            if batch_idx % args.save_interval == 0 and rank == 0:  # Print loss on rank 0
                print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        if rank == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # Save model checkpoint on rank 0
        if rank == 0 and (epoch) % 2== 0:
            checkpoint_path = f'./Models/trained_modelscheckpoint_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            save_optimizer_state(path='./Models/optimizer-ckpt.pth', rank=rank, step=epoch)
            psnr,ssim_value,eval_loss = evaluate(validation_loader)
            
            # 🐝 Log train and validation metrics to wandb
            index = {"step": epoch,
                        "total_iteration": batch_idx} 
            metrics = {"train_loss": running_loss/len(train_loader), 
                        "train_PSNR": torch.mean(psnr_list),
                        "train_SSIM": torch.mean(ssim_list)}
            val_metrics = {"val_loss": eval_loss, 
                        "val_PSNR": psnr,
                        "val_SSIM": ssim_value}
            wandb.log({**index, **metrics, **val_metrics})
            
        epoch += 1
    
    cleanup_ddp()

def evaluate(val_dataloader):
    N_in = 5
    predeblur, HR_in = False, False
    back_RBs = 7

    model = DfRes_arch_nores_nlsa.DfConv_EkSA(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in, w_TSA=False)
    
    for i, data in enumerate(val_dataloader):
        imgs_LQ, imgs_GT = data

        
        with torch.no_grad():
            
            output = model(imgs_LQ)
            
            imgs_GT = imgs_GT.permute(0, 1, 4, 2, 3)
            
            #calculate training psnr and ssim
            psnr_list = torch.zeros(output.shape[1])
            ssim_list = torch.zeros(output.shape[1])
            eval_loss = torch.zeros(output.shape[1])
            
            for batch_idx in range(output.shape[1]): #0~5
                if torch.mean((output[batch_idx] - imgs_GT[:,2,:,:,:][batch_idx]) * (output[batch_idx]- imgs_GT[:,2,:,:,:][batch_idx]))>0:
                    img_psnr =  -10 * math.log10(torch.mean((output[batch_idx] - imgs_GT[:,2,:,:,:][batch_idx]) * (output[batch_idx] - imgs_GT[:,2,:,:,:][batch_idx])))
                    img_ssim = torch.mean(ssim(output[batch_idx].unsqueeze(0).detach().cpu(),imgs_GT[:,2,:,:,:][batch_idx].unsqueeze(0).detach().cpu()))
                    img_loss = (output[batch_idx]-imgs_GT[:,2,:,:,:][batch_idx]).abs().mean()
                else:
                    img_psnr = -10 * math.log10(0.00001)
                    img_ssim = 0.00001
                    img_loss = 0.00001
                    
                psnr_list[batch_idx]=img_psnr
                ssim_list[batch_idx]=img_ssim
                eval_loss[batch_idx]=img_loss
                
    return torch.mean(psnr_list), torch.mean(ssim_list), torch.mean(eval_loss)
    
    
    
    
    
    

def main():
    parser = argparse.ArgumentParser(description='Train on Vid dataset with DDP')
    parser.add_argument('--batch-size', type=int, default=4, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--world-size', type=int, default=2, help='number of GPUs to use for DDP')
    parser.add_argument('--RESUME', type=bool, default=False, help='whether to Resume')
    parser.add_argument('--save_interval', type=int, default=30, help='interval to save')
    parser.add_argument('--ckpt_file', type=str, default='', help='chechpoint file when resume')
   
    
    args = parser.parse_args()
    
    # mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    train(rank, args.world_size, args)

if __name__ == '__main__':
    main()
