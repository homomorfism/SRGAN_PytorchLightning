import pytorch_lightning as pl
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
            GeneratorResBlock() for i in range(num_res_blocks)
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

        x_identity = nn.Identity()(x)

        x = self.model(x)
        x = self.after_training(x)
        x += x_identity

        x = self.output_model(x)

        return x
