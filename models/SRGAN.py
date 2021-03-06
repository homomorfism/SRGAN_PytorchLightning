import matplotlib.pyplot as plt
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

        self.content_loss = ContentLoss(config['vgg_coef'])
        self.adversarial_loss = AdversarialLoss()

        self.generator = Generator(num_res_blocks=config['num_res_block'])
        self.discriminator = Discriminator()

        self.train_step = 0

        self.execute_once = True

    def configure_optimizers(self):

        optim_generator = Adam(self.generator.parameters(), lr=self.config['lr'], )
        optim_discriminator = Adam(self.discriminator.parameters(), lr=self.config['lr'])

        sched_generator = StepLR(optim_generator, step_size=self.config['lr_step_size'], gamma=self.config['lr_step'])
        sched_discriminator = StepLR(optim_discriminator, step_size=self.config['lr_step_size'],
                                     gamma=self.config['lr_step'])

        return [optim_generator, optim_discriminator], [sched_generator, sched_discriminator]

    def generator_loss(self, low_resolution_image, high_resolution_image):

        discriminator_fake_outputs = self.discriminator(self.generator(low_resolution_image))
        adversarial_loss = self.adversarial_loss(discriminator_fake_outputs)
        self.log('adversarial_loss', adversarial_loss.item())

        content_loss = self.content_loss(self.generator(low_resolution_image), high_resolution_image)
        self.log('content_loss', content_loss.item())

        g_loss = content_loss + self.config['adv_loss_rate'] * adversarial_loss

        return g_loss

    def discriminator_loss(self, lr_image, hr_image):

        real_loss = torch.log(self.discriminator(hr_image)).mean()
        fake_loss = torch.log(1 - self.discriminator(self.generator(lr_image))).mean()

        return - (real_loss + fake_loss) / 2

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):

        lr_image, hr_image = batch

        if self.execute_once:
            print(f"lr shape={lr_image.shape}")  # lr shape = torch.Size([1, 3, 384, 384])
            print(f"hr shape={hr_image.shape}")  # hr shape = torch.Size([1, 3, 96, 96])
            print(f"gen shape={self.generator(lr_image).shape}")

            print(f"Range values in lr_image is ({torch.min(lr_image)}) - ({torch.max(lr_image)})")  # 0.0 - 1.0
            print(f"Range values in hr_image is ({torch.min(hr_image)}) - ({torch.max(hr_image)})")  # -1.0 - 1.0

            plt.show()

            self.execute_once = False

        # Collect log images each 100 iterations
        if optimizer_idx == 0 and self.train_step % 100 == 0:
            generated_images = self.generator(lr_image)

            generated_images_vs_source = []
            generated_images_vs_bicubic = []

            for i in range(self.config['batch_size'] // 2):
                image_crop, scale_factor = self.config['image_crop'], self.config['downsampling_factor']
                lr_shape = (image_crop // scale_factor, image_crop // scale_factor)

                bicubic = torch.nn.Upsample(scale_factor=scale_factor, mode='bicubic')

                generated_images_vs_source.append(generated_images[i])
                generated_images_vs_source.append(hr_image[i])
                generated_images_vs_bicubic.append(generated_images[i])
                generated_images_vs_bicubic.append(bicubic(lr_image)[i])

            grid_generated_images_vs_source = torchvision.utils.make_grid(
                generated_images_vs_source, normalize=True, nrow=len(generated_images_vs_source)
            )
            grid_generated_images_vs_bicubic = torchvision.utils.make_grid(
                generated_images_vs_bicubic, normalize=True, nrow=len(generated_images_vs_bicubic)
            )

            grid_lr_images = torchvision.utils.make_grid(lr_image, normalize=True, nrow=self.config['batch_size'])

            self.logger.experiment.add_image('Low resolution images : Train stage', grid_lr_images, self.train_step)
            self.logger.experiment.add_image('Generated images vs source : Train stage',
                                             grid_generated_images_vs_source, self.train_step)
            self.logger.experiment.add_image("Generated images vs bicubic lr images : Train stage",
                                             grid_generated_images_vs_bicubic, self.train_step)

        # Generator step
        if optimizer_idx == 0:
            self.train_step += 1  # Each 100 iter logs images

            gen_loss = self.generator_loss(lr_image, hr_image)

            self.log('generator_loss', gen_loss.item())
            return gen_loss

        # Discriminator loss
        else:
            disc_loss = self.discriminator_loss(lr_image, hr_image)

            self.log('discriminator_loss', disc_loss.item())
            return disc_loss
