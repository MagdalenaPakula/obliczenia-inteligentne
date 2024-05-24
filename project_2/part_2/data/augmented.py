import pytorch_lightning as pl
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

from project_2.part_2.data import DATA_DIR


class _AugmentedDataModule(pl.LightningDataModule):
    def __init__(self, subset_size, augmentation, num_augmentations: int):
        super().__init__()
        self.subset_size = subset_size
        self.augmentation = augmentation
        self.num_augmentations = num_augmentations

        type Dataset = VisionDataset | TensorDataset | None
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None

    @property
    def _batch_size(self):
        # minimum 500 batches
        batches_500 = int(self.dataset_size / 500)
        return min(64, max(batches_500, 1))

    @property
    def dataset_size(self) -> int:
        return self.subset_size * self.num_augmentations

    def _augment_dataset(self, train_dataset):
        classes: torch.Tensor = torch.tensor(train_dataset.targets, dtype=torch.int).unique()
        num_classes: int = classes.size(0)
        num_samples_single_class = self.subset_size // num_classes

        generator = torch.Generator()
        augmented_images: torch.Tensor | None = None
        labels = torch.zeros(0)

        for class_label in classes:
            # Get indices of data points corresponding to that class
            # noinspection PyUnresolvedReferences
            indices = (torch.tensor(train_dataset.targets, dtype=torch.int) == class_label).nonzero().squeeze(1)
            # select only needed amount of random indices
            indices = indices[torch.randperm(len(indices), generator=generator)[:num_samples_single_class]]

            for index_tensor in indices:
                index = index_tensor.item()

                # get item of the selected index from the dataset
                image, label = train_dataset[index]

                # append the label
                label = torch.tensor([label for _ in range(self.num_augmentations)])
                labels = torch.cat((labels, label))

                for _ in range(self.num_augmentations):
                    # apply the augmentation to the data
                    augmented_sample = self.augmentation(image)

                    # append the transformed image
                    if augmented_images is not None:
                        augmented_images = torch.cat((augmented_images, augmented_sample.unsqueeze(0)))
                    else:
                        augmented_images = augmented_sample.unsqueeze(0)

        assert augmented_images is not None
        assert augmented_images.size(0) == labels.size(0)

        return augmented_images, labels

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        if batch_size is None:
            batch_size = self._batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        if batch_size is None:
            batch_size = int(self.dataset_size/10)
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


class AugmentedMNISTDataModule(_AugmentedDataModule):
    """
    Creates an augmented MNIST data module by applying a specified augmentations on the randomly chosen
    representative sample of the original MNIST dataset.

    The final size of the dataset
    will be :paramref:`num_augmentations` * :paramref:`subset_size`.

    :param transform: Tranform to be applied to the whole dataset.
    :param subset_size: Number of images that should be taken from the original dataset.
    :param augmentation: Transform to apply augmentations to the image. Will not be applied to the test set.
    :param num_augmentations: Number of times to apply augmentation to a single image.
    """

    def __init__(self,
                 transform: v2.Transform,
                 subset_size: int,
                 augmentation: v2.Transform,
                 num_augmentations: int,
                 ):
        super().__init__(subset_size, augmentation, num_augmentations)
        self.transform = transform
        self.data_dir = DATA_DIR

    def setup(self, stage=None):
        if self.test_dataset is None:
            # test dataset should have no augmentations applied
            self.test_dataset = MNIST(self.data_dir, download=True, train=False, transform=self.transform)

        if self.train_dataset is None:
            train_dataset = MNIST(self.data_dir, download=True, train=True, transform=self.transform)

            augmented_images, labels = self._augment_dataset(train_dataset)
            self.train_dataset = TensorDataset(augmented_images, labels)


class AugmentedCIFAR10DataModule(_AugmentedDataModule):
    """
    Creates an augmented CIFAR10 data module by applying a specified augmentations on the randomly chosen
    representative sample of the original MNIST dataset.

    The final size of the dataset
    will be :paramref:`num_augmentations` * :paramref:`subset_size`.

    :param transform: Tranform to be applied to the whole dataset.
    :param subset_size: Number of images that should be taken from the original dataset.
    :param augmentation: Transform to apply augmentations to the image. Will not be applied to the test set.
    :param num_augmentations: Number of times to apply augmentation to a single image.
    """

    def __init__(self,
                 transform: v2.Transform,
                 subset_size: int,
                 augmentation: v2.Transform,
                 num_augmentations: int,
                 ):
        super().__init__(subset_size, augmentation, num_augmentations)
        self.transform = transform
        self.data_dir = DATA_DIR

    def setup(self, stage=None):
        if self.test_dataset is None:
            # test dataset should have no augmentations applied
            self.test_dataset = CIFAR10(self.data_dir, download=True, train=False, transform=self.transform)

        if self.train_dataset is None:
            train_dataset = CIFAR10(self.data_dir, download=True, train=True, transform=self.transform)

            augmented_images, labels = self._augment_dataset(train_dataset)
            self.train_dataset = TensorDataset(augmented_images, labels)


def _show_augmented_mnist():
    torch.manual_seed(42)

    mnist = MNIST(DATA_DIR, train=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                  download=False)

    aug = AugmentedMNISTDataModule(
        transform=v2.Compose([v2.ToTensor()]),
        augmentation=v2.RandomAffine(degrees=10, scale=(0.6, 1.5), shear=20),
        subset_size=20,
        num_augmentations=5,
    )
    aug.setup()

    train_dl = aug.train_dataloader(batch_size=25)
    images, labels = next(iter(train_dl))

    grid = torchvision.utils.make_grid(images, nrow=5, padding=3, pad_value=1.)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    _show_augmented_mnist()
