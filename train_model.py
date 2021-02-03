import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_loader import ImageNetDataLoader
from models.SRGAN import SRGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/conf.yaml')
    parser.add_argument('--resume', help='resume from the latest checkpoint', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_callback = ModelCheckpoint(
        filepath=config['checkpoints_folder'],
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='generator_loss',
        mode='min'
    )

    logger = TensorBoardLogger(
        save_dir=config['tensorboard_log_dir'],
        name='sngan'
    )

    train_loader = ImageNetDataLoader(
        data_folder=config['data_folder'],
        image_crop=config['image_crop'],
        downsampling_factor=config['downsampling_factor'],
        batch_size=config['batch_size']
    ).train_dataloader()

    model = SRGAN(config=config)

    if args.resume:
        print("Resuming training from the latest checkpoint...")

        trainer = pl.Trainer(
            gpus=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=os.path.join(config['checkpoints_folder'], 'last.ckpt')
        )

    else:
        print("Starting training...")

        trainer = pl.Trainer(
            gpus=1,
            logger=logger,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
