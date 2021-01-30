import pytorch_lightning as pl
import torch.nn as nn


class GeneratorResBlock(pl.LightningModule):
    def __init__(self):
        super(GeneratorResBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )

    def forward(self, x):
        # Change to .clone()
        tmp = nn.Identity()(x)
        output = self.model(x)

        return output + tmp


class DiscriminatorResBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DiscriminatorResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.model(x)
