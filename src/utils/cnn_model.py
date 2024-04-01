
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,1,1),
            nn.ReLU(),
            nn.Conv2d(1, 32, 3, padding=1),  # batch x 32 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),  # batch x 32 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),  # batch x 64 x 128 x 128
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 128 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 128 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.ReLU(), # Consider using nn.Sigmoid() if the final output needs to be in [0,1]
            nn.Conv2d(1,3,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # Before passing to decoder, ensure the tensor is reshaped or flattened as needed
        x = self.decoder(x)
        return x


# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder,self).__init__()
#         self.encoder=nn.Sequential(
#             nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#             nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2)
#         )
#         self.decoder=nn.Sequential(
#             nn.ConvTranspose2d(8,16,kernel_size=3,stride=2,padding=1,output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16,3,kernel_size=3,stride=2,padding=1,output_padding=1),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         x=self.encoder(x)
#         x=self.decoder(x)
#         return x

# import torch
# import torch.nn as nn
# import math

# class Autoencoder(nn.Module):

# 	def __init__(self):
# 		super(Autoencoder, self).__init__()
		
# 		# LightDehazeNet Architecture 
# 		self.relu = nn.ReLU(inplace=True)

# 		self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
# 		self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
# 		self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 
# 		self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True) 
# 		self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
# 		self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
# 		self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
# 		self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)
		
# 	def forward(self, img):
# 		pipeline = []
# 		pipeline.append(img)

# 		conv_layer1 = self.relu(self.e_conv_layer1(img))
# 		conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
# 		conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

# 		# concatenating conv1 and conv3
# 		concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)
		
# 		conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
# 		conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
# 		conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

# 		# concatenating conv4 and conv6
# 		concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)
		
# 		conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

# 		# concatenating conv2, conv5, and conv7
# 		concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)
		
# 		conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


# 		dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
# 		#J(x) = clean_image, k(x) = x8, I(x) = x, b = 1
		
		
# 		return dehaze_image 


