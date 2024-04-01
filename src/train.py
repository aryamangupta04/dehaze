import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from utils.parse import opt
from utils.cnn_model import Autoencoder
from utils.dataset_utils import OTS_train_loader,ITS_train_loader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def train(train_loader,model,epochs,iterations,device):
    i=0
    model.train()
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
        if(epoch==3):
            validate_and_visualize(val_loader,model,device)
        i=i+1
def validate_and_visualize(val_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed for validation
        for data, targets in val_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            outputs = model(data)
            visualize_sample(data.cpu(), outputs.cpu(), targets.cpu())
            break

def visualize_sample(original, generated, target):
    # Inverse Normalize for visualization if your data is normalized
    inv_normalize = transforms.Normalize(
        mean=[-0.64/0.14, -0.6/0.15, -0.58/0.152],
        std=[1/0.14, 1/0.15, 1/0.152]
    )
    grid = make_grid([original, generated, target])
    plt.figure(figsize=(12, 4))
    plt.imshow(transforms.ToPILImage()(grid))
    plt.axis('off')
    plt.title("Original - Generated - Target")
    plt.show()

if __name__=='__main__':
    train_loader=OTS_train_loader
    model=Autoencoder().to(opt.device)
    loss_func=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=opt.lr)
    val_loader=ITS_train_loader
    train(train_loader,model,opt.epochs,opt.itr,opt.device)
    
    
