import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from utils.parse import opt
from utils.cnn_model import Autoencoder
from utils.dataset_utils import OTS_train_loader
def train(train_loader,model,epochs,iterations,device):
    i=0
    for epoch in range(epochs):
        epoch_loss=0.0
        t=0
        for data,targets in train_loader:
            data=data.to(device=device)
            targets=targets.to(device=device)
            x=model(data)
            loss=loss_func(x,targets)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss+=loss.item()
        print("the loss for epoch ",i," is ",epoch_loss)
        i=i+1


if __name__=='__main__':
    train_loader=OTS_train_loader
    model=Autoencoder().to(opt.device)
    loss_func=nn.L1Loss()
    optimizer=optim.Adam(model.parameters(),lr=opt.lr)
    train(train_loader,model,opt.epochs,opt.itr,opt.device)
    
    
