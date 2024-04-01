
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(8,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
        )

        # Dynamically determine the output size of the convolutional layers
        with torch.no_grad():
            self.flatten_size = self._get_conv_output([1, 3, 550, 413])

        # Define the linear layer using the dynamically determined size
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 1),
            nn.Sigmoid(),
        )

    def _get_conv_output(self, shape):
        dummy_input = torch.autograd.Variable(torch.zeros(shape))
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             # Flatten before going to linear layer
#             nn.Flatten(),
#             # Adjust the following layer to match the output of the last Conv2d layer
#             nn.Linear(64 * 78 * 58, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x
