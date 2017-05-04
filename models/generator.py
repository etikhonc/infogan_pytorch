import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, image_shape, network_type='mnist'):
        super(Generator, self).__init__()

        self.image_size = image_shape[0:2]  # (H, W)

        self.fc1 = torch.nn.Linear(74, 1024)
        self.batchNorm1 = torch.nn.BatchNorm1d(1024, momentum=0.1)
        self.fc2 = torch.nn.Linear(1024, self.image_size[0]/4 * self.image_size[1]/4 * 128)
        self.batchNorm2 = torch.nn.BatchNorm1d(self.image_size[0]/4 * self.image_size[1]/4 * 128, momentum=0.1)
        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batchNorm3 = torch.nn.BatchNorm2d(64, momentum=0.1)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z_var):
        x = F.relu(self.batchNorm1(self.fc1(z_var)))
        x = F.relu(self.batchNorm2(self.fc2(x)))
        # reshape to the 128xHxW
        x = x.view(x.size(0), 128, self.image_size[0]/4, self.image_size[0]/4)
        x = F.relu(self.batchNorm3(self.deconv1(x)))
        x = self.deconv2(x)
        # flatten
        g_out = x.view(x.size(0), np.prod(x.size()[1:]))
        return g_out


