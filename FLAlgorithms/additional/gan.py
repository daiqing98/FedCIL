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
        self.fc_aux = nn.Linear((image_size//8)**2 * channel_size*4, self.num_classes)
         
        # ReACGAN
        d_embed_dim = 512
        self.embedding = nn.Linear(self.num_classes, d_embed_dim)
        self.linear2 = nn.Linear((image_size//8)**2 * channel_size*4, d_embed_dim)
        
        
    def forward(self, x, if_features=False):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # flat
        x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
        features = x
        
        classes = self.fc_aux(x)
        logits = classes
        realfake = self.fc(x)
        
        classes_p = self.softmax(classes)
        realfake = self.sigmoid(realfake).squeeze()
        
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