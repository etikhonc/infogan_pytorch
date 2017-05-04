import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, image_shape, dist_flat_dim):
        super(Discriminator, self).__init__()

        self.image_shape = image_shape
        self.shared_template_part1 = nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128, momentum=0.1),
            torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.shared_template_part2 = nn.Sequential(
            torch.nn.Linear(6272, 1024),
            torch.nn.BatchNorm1d(1024, momentum=0.1),
            torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.discriminator_end = nn.Linear(1024, 1)

        self.encoder_end = nn.Sequential(
            torch.nn.Linear(1024, 128),
            torch.nn.BatchNorm1d(128, momentum=0.1),
            torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
            torch.nn.Linear(128, dist_flat_dim)
        )

    def forward(self, x_var):
        x = x_var.view(x_var.size(0), 1, self.image_shape[0], self.image_shape[1])
        x = self.shared_template_part1(x)
        # x: 128x128x7x7
        x = x.view(x.size(0), np.prod(x.size()[1:]))
        x = self.shared_template_part2(x)

        d_out = self.discriminator_end(x)
        e_out = self.encoder_end(x)

        return d_out, e_out


