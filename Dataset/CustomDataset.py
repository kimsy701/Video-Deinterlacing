import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import Dataset.utils as util
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataset_name, lq_folder, gt_folder, lq_folder_val, gt_folder_val, data_mode=None, transform=None):
        
        self.dataset_name = dataset_name
        
        if self.dataset_name=="train":
            self.lq_folder=lq_folder
            self.gt_folder=gt_folder
        elif self.dataset_name=="validation":
            self.lq_folder=lq_folder_val
            self.gt_folder=gt_folder_val
            
        self.data_mode = data_mode
        self.transform = transform
        

        self.lq_subfolders = sorted(glob.glob(os.path.join(self.lq_folder, '*')))
        self.gt_subfolders = sorted(glob.glob(os.path.join(self.gt_folder, '*')))
        


    def __len__(self):
        return len(self.lq_subfolders)

    def __getitem__(self, idx):
        print("idx", idx)
        lq_subfolder = self.lq_subfolders[idx]
        gt_subfolder = self.gt_subfolders[idx]

        lq_img_paths = sorted(glob.glob(os.path.join(lq_subfolder, '*')))
        gt_img_paths = sorted(glob.glob(os.path.join(gt_subfolder, '*')))

        lq_imgs,img_path_l = util.read_img_seq(lq_img_paths)
        # print("lq_imgs[2] avg",torch.mean(lq_imgs[2])) #서로 다른 값으로, 0아닌 값으로 나옴
        gt_imgs = [torch.Tensor(util.read_img(None, img_path)) for img_path in gt_img_paths]
        # print("len(gt_imgs)",len(gt_imgs)) #5
        # print(gt_imgs[0].shape) #torch.Size([256, 448, 3])


        if self.data_mode == 'evenodd':
            c, h, w = lq_imgs[0].shape
            for i in range(len(lq_imgs)):
                if i % 2 == 0:
                    # lq_imgs[i] = np.zeros((c, h, w), dtype=lq_imgs[i].dtype)
                    # print("lq_imgs[i].dtype",lq_imgs[i].dtype) #torch.float32
                    lq_imgs[i] = torch.zeros((c, h, w), dtype=torch.float32)
                else:
                    # lq_imgs[i] = np.ones((c, h, w), dtype=lq_imgs[i].dtype)
                    lq_imgs[i] = torch.ones((c, h, w), dtype=torch.float32)
            # print("sum(lq_imgs)",sum(lq_imgs))       

        if self.transform:
            print("self transform 들어감") #transform : 디폴트 값이 None임으로 안들어감...
            lq_imgs = self.transform(lq_imgs)
            gt_imgs = [torch.from_numpy(self.transform(img)) for img in gt_imgs]

        # print("torch.stack(gt_imgs) shape",torch.stack(gt_imgs).shape) #torch.Size([5, 256, 448, 3])
        # print("lq_imgs[2] avg",torch.mean(lq_imgs[2]))  #다 0됨 
        return lq_imgs, torch.stack(gt_imgs),img_path_l

# if __name__ == '__main__':
#     lq_folder = './Dataset/train_val_100'
#     gt_folder = './Dataset/gt_val_100_re'

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     dataset = CustomDataset(lq_folder, gt_folder, data_mode='evenodd', transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

#     for lq_imgs, gt_imgs in dataloader:
#         print("here in dataloader", lq_imgs.shape, gt_imgs.shape)
#         break
