from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10

_DATA_DIR = str(Path(__file__).parent.parent / 'data')


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 100
        self.data_dir = _DATA_DIR
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage: str) -> None:
        self.train_dataset: Dataset = MNIST(self.data_dir, train=True, transform=self.transform)
        self.test_dataset: Dataset = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.data_dir = _DATA_DIR
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage: str) -> None:
        self.train_dataset: Dataset = CIFAR10(self.data_dir, train=True, download=True)
        self.test_dataset: Dataset = CIFAR10(self.data_dir, train=False, download=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
