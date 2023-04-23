from torch import nn
from torch.nn import functional as F
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch

class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size, num_classes = 10):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size
        
        self.num_classes = num_classes
        
        # activation functions:
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        
        self.bn0 = nn.BatchNorm2d(channel_size)
        
        self.conv2 = nn.Conv2d(
            channel_size, channel_size*2,
            kernel_size=4, stride=2, padding=1
        )
        
        self.bn1 = nn.BatchNorm2d(channel_size*2)
        
        self.conv3 = nn.Conv2d(
            channel_size*2, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
                
        self.bn2 = nn.BatchNorm2d(channel_size*4)
        
        self.fc = nn.Linear((image_size//8)**2 * channel_size*4, 1)
        
        # aux-classifier fc
        self.fc_aux = nn.Linear((image_size//8)**2 * channel_size*4, self.num_classes)
        
    def forward(self, x, if_features=False):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
        features = x
        
        classes = self.fc_aux(x)
        realfake = self.fc(x)
 
        logits = classes
        
        classes_p = self.softmax(classes)
        realfake = self.sigmoid(realfake).squeeze()
        
        #return realfake, classes, logits
        
        ### If WGAN: realfake are not activated! ###
        if if_features==False :
            return realfake, classes_p, logits
        else:
             return realfake, classes_p, logits, features
        
class Generator(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.fc = nn.Linear(z_size, (image_size//8)**2 * channel_size*8)
        self.bn0 = nn.BatchNorm2d(channel_size*8)
        self.bn1 = nn.BatchNorm2d(channel_size*4)
        
        self.deconv1 = nn.ConvTranspose2d(
            channel_size*8, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size*2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size*4, channel_size*2,
            kernel_size=4, stride=2, padding=1,
        )
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size*2, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0),
            self.channel_size*8,
            self.image_size//8,
            self.image_size//8,
        )))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        
        # according to MemReplay GAN & AC-GAN, activation(it is tanh at that case) is applied.
        return F.tanh(g)
    
# ============================== CIFAR-10 ============================

class Generator_CIFAR(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        super().__init__()
        self.nz = z_size

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 384, 1, 1)  
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5

        return output


class Critic_CIFAR(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size):
        super(Critic_CIFAR, self).__init__()
        num_classes=10
        
        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
      
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 4*4*512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        logits = fc_aux
        
        # sigmoided, softmaxed
        return realfake, classes, logits
