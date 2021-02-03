import pytorch_lightning as pl
import torch
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.Discriminator import Discriminator
from models.Generator import Generator
from models.utils import ContentLoss, AdversarialLoss


class SRGAN(pl.LightningModule):

    def __init__(self, config):

        super(SRGAN, self).__init__()
        self.config = config

        print(f"config={config}")

        self.content_loss = ContentLoss()
        self.adversarial_loss = AdversarialLoss()

        self.generator = Generator(num_res_blocks=config['num_res_block'])
        self.discriminator = Discriminator()

        self.train_step = 0

    def configure_optimizers(self):

        optim_generator = Adam(self.generator.parameters(), lr=self.config['lr'])
        optim_discriminator = Adam(self.discriminator.parameters(), lr=self.config['lr'])

        sched_generator = StepLR(optim_generator, step_size=self.config['lr_step_size'], gamma=self.config['lr_step'])
        sched_discriminator = StepLR(optim_discriminator, step_size=self.config['lr_step_size'],
                                     gamma=self.config['lr_step'])

        return [optim_generator, optim_discriminator], [sched_generator, sched_discriminator]

    def generator_loss(self, low_resolution_image, high_resolution_image):

        g_loss = self.content_loss(low_resolution_image, high_resolution_image) + \
                 self.config['adv_loss_rate'] * self.adversarial_loss(self.generator, self.discriminator,
                                                                      low_resolution_image)

        return g_loss

    def discriminator_loss(self, lr_image, hr_image):

        real_loss = torch.log(self.discriminator(hr_image))
        fake_loss = torch.log(1 - self.discriminator(self.generator(lr_image)))

        return (real_loss + fake_loss) / 2

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):

        lr_image, hr_image = batch

        # Collect log images each 100 iterations
        if optimizer_idx == 0 and self.train_step % 100 == 0:
            generated_images = self.generator(lr_image)

            grid_source_images = torchvision.utils.make_grid(hr_image, normalize=True, nrow=4)
            grid_lr_images = torchvision.utils.make_grid(lr_image, normalize=True, nrow=4)
            grid_generated_images = torchvision.utils.make_grid(generated_images, normalize=True, nrow=4)

            self.logger.experiment.add_image('Real images: Train stage', grid_source_images, self.train_step)
            self.logger.experiment.add_image('Generated images : Train stage', grid_generated_images, self.train_step)
            self.logger.experiment.add_image('Low resolution images : Train stage', grid_lr_images, self.train_step)

        # Generator step
        if optimizer_idx == 0:
            gen_loss = self.generator_loss(lr_image, hr_image)
            self.log('gen_loss', gen_loss.item())
            return gen_loss

        # Discriminator loss
        else:
            disc_loss = self.discriminator_loss(lr_image, hr_image)
            self.log('disc_loss', disc_loss.item())
            return disc_loss
