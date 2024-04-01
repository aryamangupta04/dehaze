import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from utils.parse import opt
from utils.cnn_model import Autoencoder
from utils.dataset_utils import OTS_train_loader,OTS_val_loader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.utils
import numpy as np
def show_images(images, title=None, nrow=5):
    """
    Utility function for showing images with matplotlib
    """
    images = torchvision.utils.make_grid(images, nrow=nrow)
    np_images = images.numpy()
    plt.figure(figsize=(20, 10))
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
def train(train_loader,model,epochs,iterations,device):
    i=0
    model.train()
    highest_psnr=0.0
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
        # if(epoch==3):
        #     validate_and_visualize(val_loader,model,device)
        i=i+1
        avg_psnr = validate_and_calculate_psnr(val_loader, model, device)
        print(f"Epoch {epoch + 1}, Average PSNR: {avg_psnr} dB")
        t=t+1
        if t==1:
            show_images(data.cpu(), title="Input Images", nrow=5)
            show_images(x.cpu(), title="Generated Images", nrow=5)
            show_images(targets.cpu(), title="Target Images", nrow=5)

        # Save the model if it has the highest PSNR so far
        if avg_psnr > highest_psnr:
            highest_psnr = avg_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved better model with PSNR: {highest_psnr} dB")
def validate_and_calculate_psnr(val_loader, model, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            mse_loss = nn.MSELoss()(outputs, targets)
            psnr = 20 * torch.log10(1 / mse_loss)
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)
    return avg_psnr
# def validate_and_visualize(val_loader, model, device):
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():  # No gradients needed for validation
#         for data, targets in val_loader:
#             data = data.to(device=device)
#             targets = targets.to(device=device)
#             outputs = model(data)
#             visualize_sample(data.cpu(), outputs.cpu(), targets.cpu())
#             break

# def visualize_sample(original, generated, target):
#     # Inverse Normalize for visualization if your data is normalized
#     inv_normalize = transforms.Normalize(
#         mean=[-0.64/0.14, -0.6/0.15, -0.58/0.152],
#         std=[1/0.14, 1/0.15, 1/0.152]
#     )
#     grid = make_grid([original, generated, target])
#     plt.figure(figsize=(12, 4))
#     plt.imshow(transforms.ToPILImage()(grid))
#     plt.axis('off')
#     plt.title("Original - Generated - Target")
#     plt.show()

if __name__=='__main__':
    train_loader=OTS_train_loader
    model=Autoencoder().to(opt.device)
    loss_func=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=opt.lr)
    val_loader=OTS_val_loader
    train(train_loader,model,opt.epochs,opt.itr,opt.device)
    
    
