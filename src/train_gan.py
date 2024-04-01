import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from utils.parse import opt
from utils.gan_model import Generator,Discriminator
from utils.dataset_utils import OTS_train_loader
def train_GAN(generator, discriminator, dataloader, num_epochs):
    # Losses & optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss_g=0.0
        epoch_loss_d=0.0
        for data,imgs in dataloader:
            valid = torch.ones((imgs.size(0), 1), requires_grad=False)
            fake = torch.zeros((imgs.size(0), 1), requires_grad=False)

            # Train Generator
            optimizer_G.zero_grad()
            #z = torch.randn(imgs.shape[0], 3, 620, 460) # Random noise
            generated_imgs = generator(data)
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            epoch_loss_g+=g_loss.item()
            epoch_loss_d+=d_loss.item()
            #print(f"Epoch: {epoch}, Batch: {i}, D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")
        print("the generator loss for the epoch is ",epoch_loss_g," the discriminator loss for the epoch is ",epoch_loss_d)
        

# Initialize models
generator = Generator()
discriminator = Discriminator()
dataloader=OTS_train_loader
num_epochs=opt.epochs
train_GAN(generator,discriminator,dataloader,num_epochs)


