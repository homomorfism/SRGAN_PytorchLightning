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
