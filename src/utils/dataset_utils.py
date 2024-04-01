import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
from torchvision.transforms import Resize

sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader,random_split
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from utils.parse import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

def tensorShow(tensors,titles=None):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
        self.resize = Resize((460, 620))
        print(os.path)
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        haze=self.resize(haze)
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(1400,1450)
                haze=Image.open(self.haze_imgs[index])
                haze=self.resize(haze)
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        id=id.split('\\')[-1]
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=self.resize(clear)
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        # haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        haze=tfs.ToTensor()(haze)
        clear=tfs.ToTensor()(clear)
        return haze,clear
    # def augData(self,data,target):
    #     if self.train:
    #         rand_hor=random.randint(0,1)
    #         rand_rot=random.randint(0,3)
    #         data=tfs.RandomHorizontalFlip(rand_hor)(data)
    #         target=tfs.RandomHorizontalFlip(rand_hor)(target)
    #         if rand_rot:
    #             data=FF.rotate(data,90*rand_rot)
    #             target=FF.rotate(target,90*rand_rot)
    #     data=tfs.ToTensor()(data)
    #     data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
    #     target=tfs.ToTensor()(target)
    #     return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

import os
pwd=os.getcwd()
print(pwd)
#path='C:/Users/HP PAVILION/Desktop/gans/dataset'
path='/kaggle/working/dehaze/dataset'
#path to your 'data' folder
OTS_dataset = RESIDE_Dataset(path+'/SOTS/outdoor', train=False, size='whole img')

# Define split sizes
total_size = len(OTS_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Randomly split the dataset
OTS_train_dataset, OTS_val_dataset, OTS_test_dataset = random_split(OTS_dataset, [train_size, val_size, test_size])

# Create DataLoader for each set
OTS_train_loader = DataLoader(dataset=OTS_train_dataset, batch_size=16, shuffle=True)
OTS_val_loader = DataLoader(dataset=OTS_val_dataset, batch_size=1, shuffle=False)
OTS_test_loader = DataLoader(dataset=OTS_test_dataset, batch_size=1, shuffle=False)

# ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/SOTS/indoor',train=True,size=crop_size),batch_size=1,shuffle=True)
# OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/SOTS/outdoor',train=False,size='whole img'),batch_size=16,shuffle=False)

#OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
#OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

# for batch in ITS_train_loader:
#     haze_images, clear_images = batch
#     # Display the images
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].imshow(make_grid(haze_images, nrow=BS).permute(1, 2, 0))
#     axes[0].set_title('Haze Images')
#     axes[0].axis('off')
#     axes[1].imshow(make_grid(clear_images, nrow=BS).permute(1, 2, 0))
#     axes[1].set_title('Clear Images')
#     axes[1].axis('off')
#     plt.show()
#     break
if __name__ == "__main__":
    pass