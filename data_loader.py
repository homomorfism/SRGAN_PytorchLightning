import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class ImageNetDataset(Dataset):
    def __init__(self, data_folder, image_crop, downsampling_factor):
        super(ImageNetDataset, self).__init__()

        self.data_folder = data_folder

        # [-1, 1]
        self.hr_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(size=(image_crop, image_crop)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.lr_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(size=(image_crop, image_crop)),
            transforms.Resize(image_crop // downsampling_factor),
            transforms.ToTensor(),
        ])

        self.dataset = datasets.ImageFolder(
            root=data_folder
        )

    def __getitem__(self, item):
        image, label = self.dataset[item]

        hr = self.hr_transform(image)
        lr = self.lr_transform(image)

        return lr, hr

    def __len__(self):
        return len(self.dataset)


class ImageNetDataLoader(pl.LightningDataModule):

    def __init__(self, data_folder, image_crop, downsampling_factor, batch_size):
        super(ImageNetDataLoader, self).__init__()

        self.data_folder = data_folder
        self.image_crop = image_crop
        self.downsampling_factor = downsampling_factor
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ImageNetDataset(self.data_folder, self.image_crop, self.downsampling_factor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
