import pytorch_lightning as pl
import torch.nn as nn
from torch.functional import F
from torchvision.models import vgg19


class GeneratorResBlock(pl.LightningModule):
    def __init__(self, channels):
        super(GeneratorResBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels),
        )

    def forward(self, x):
        output = self.model(x)

        return output + x


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


class ContentLoss(pl.LightningModule):
    # Add pixel-wise loss
    def __init__(self):
        super(ContentLoss, self).__init__()

        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:36]).eval()
        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, input_tensor, target_tensor):
        # print(f"input feature shape={input_tensor.shape}")
        # print(f"target image shape={target_tensor.shape}")

        return F.mse_loss(
            self.features(input_tensor),
            self.features(target_tensor)
        )


class AdversarialLoss(pl.LightningModule):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, discriminator_fake_outputs):
        return - discriminator_fake_outputs.log().sum()
