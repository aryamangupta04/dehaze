import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from utils.parse import opt
from utils.gan_model import Generator,Discriminator
from utils.dataset_utils import OTS_train_loader,OTS_val_loader

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

def train_GAN(generator, discriminator, train_loader, val_loader, device, num_epochs):
    # Losses & optimizers
    lam=0.1
    adversarial_loss = nn.BCELoss()
    rec_loss=nn.MSELoss()
    g_lr=0.002
    optimizer_G = optim.Adam(generator.parameters(), lr=0.03)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0009)
    highest_psnr = 0.0
    for epoch in range(num_epochs):
        t=0
        generator.train()
        discriminator.train()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        for data,targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            generated_imgs = generator(data)
            valid = torch.ones((targets.size(0), 1), device=device, requires_grad=False)
            fake = torch.zeros((targets.size(0), 1), device=device, requires_grad=False)
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(targets), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            # generated_imgs = generator(data)
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_rec_loss=rec_loss(generated_imgs,targets)
            g_total_loss=(lam*g_loss)+g_rec_loss
            g_total_loss.backward()
            optimizer_G.step()

            # Train Discriminator


            epoch_loss_g += g_total_loss.item()
            epoch_loss_d += d_loss.item()
            t=t+1
            # if t==1:
            #     show_images(data.cpu(), title="Input Images", nrow=5)
            #     show_images(generated_imgs.cpu(), title="Generated Images", nrow=5)
            #     show_images(targets.cpu(), title="Target Images", nrow=5)

        avg_psnr = validate_and_calculate_psnr(val_loader, generator, device)
        print(f"Epoch {epoch + 1}, G_loss: {epoch_loss_g:.4f}, D_loss: {epoch_loss_d:.4f}, Avg PSNR: {avg_psnr:.2f} dB")
        if avg_psnr > highest_psnr:
            highest_psnr = avg_psnr
            torch.save(generator.state_dict(), 'best_generator_model.pth')
            print(f"Saved better generator model with PSNR: {highest_psnr:.2f} dB")

def validate_and_calculate_psnr(val_loader, generator, device):
    generator.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = generator(data)
            mse_loss = nn.MSELoss()(outputs, targets)
            psnr = 20 * torch.log10(1 / mse_loss)
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)
    return avg_psnr

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train_loader = OTS_train_loader
    val_loader = OTS_val_loader
    num_epochs = opt.epochs
    train_GAN(generator, discriminator, train_loader, val_loader, device, num_epochs)

# def train_GAN(generator, discriminator, dataloader, num_epochs):
#     # Losses & optimizers
#     adversarial_loss = nn.BCELoss()
#     optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

#     for epoch in range(num_epochs):
#         epoch_loss_g=0.0
#         epoch_loss_d=0.0
#         for data,imgs in dataloader:
#             valid = torch.ones((imgs.size(0), 1), requires_grad=False)
#             fake = torch.zeros((imgs.size(0), 1), requires_grad=False)

#             optimizer_G.zero_grad()
#             generated_imgs = generator(data)
#             g_loss = adversarial_loss(discriminator(generated_imgs), valid)
#             g_loss.backward()
#             optimizer_G.step()

#             # Train Discriminator
#             optimizer_D.zero_grad()
#             real_loss = adversarial_loss(discriminator(imgs), valid)
#             fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
#             d_loss = (real_loss + fake_loss) / 2
#             d_loss.backward()
#             optimizer_D.step()
#             epoch_loss_g+=g_loss.item()
#             epoch_loss_d+=d_loss.item()
#             #print(f"Epoch: {epoch}, Batch: {i}, D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")
#         print("the generator loss for the epoch is ",epoch_loss_g," the discriminator loss for the epoch is ",epoch_loss_d)
        

# # Initialize models
# generator = Generator()
# discriminator = Discriminator()
# dataloader=OTS_train_loader
# num_epochs=opt.epochs
# train_GAN(generator,discriminator,dataloader,num_epochs)


