import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.utils import DiscriminatorResBlock


class Discriminator(pl.LightningModule):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.preprocess_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # padding = 1 by default
        self.features = nn.Sequential(
            DiscriminatorResBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            DiscriminatorResBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            DiscriminatorResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            DiscriminatorResBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            DiscriminatorResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            DiscriminatorResBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            DiscriminatorResBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        )

        self.classificator = nn.Sequential(

            # idk, (-1, 512, 28, 28) -> (-1, 512, 1, 1),
            # it is right ?
            nn.AdaptiveAvgPool2d(1),

            # The same as Linear
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.preprocess_layer(x)
        x = self.features(x)
        x = self.classificator(x)

        return torch.sigmoid(x.view(batch_size))
