
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 3, 3, stride=1, padding=1)
        
        self.tan = nn.Tanh()

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        t4 = out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t4)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        return self.tan(out)
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(1, 32, 3, padding=1),  # batch x 32 x 256 x 256
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, 3, padding=1),  # batch x 32 x 256 x 256
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 256 x 256
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 256 x 256
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2,2),  # batch x 64 x 128 x 128
#             nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 128 x 128
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 128 x 128
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 64 x 64
#             nn.ReLU(),
#             nn.BatchNorm2d(256)
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128, 128, 3, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64, 64, 3, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.ConvTranspose2d(32, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
#             nn.ReLU(), # Consider using nn.Sigmoid() if the final output needs to be in [0,1]
#             nn.Conv2d(1,3,1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         # Before passing to decoder, ensure the tensor is reshaped or flattened as needed
#         x = self.decoder(x)
#         return x


