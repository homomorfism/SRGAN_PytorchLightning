Hello, this ia implementation of SRGAN in pytorch_lightning.

Prepare data:

- put CelebA archive into folder data and ```unzip``` it (subfolder does not matter)

Run tensorboard for collecting logs:

- ```bash nohup tensorboard --logdir "log_dir" --port 6006 --bind_all &```

To run program:

- from a checkpoint: ```bash clear && nohup python train_model.py --resume checkpoints/last.ckpt```
- from a scratch: ```bash clear && nohup python train_model.py &```

Temp pictures (from training):

![pic](images/pic.jpg)

