import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.utils import GeneratorResBlock


class Generator(pl.LightningModule):
    def __init__(self, num_res_blocks):
        super(Generator, self).__init__()

        self.preprocess_data = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.model = nn.Sequential(*[
            GeneratorResBlock(channels=64) for i in range(num_res_blocks)
        ])

        self.after_training = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )

        self.output_model = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        x = self.preprocess_data(x)

        output = self.model(x)
        output = self.after_training(output)

        output = self.output_model(x + output)

        return (torch.tanh(output) + 1) / 2
