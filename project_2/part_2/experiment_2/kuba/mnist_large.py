import pytorch_lightning as pl
from torchvision.transforms import v2

from project_2.part_2.data.augmented import AugmentedMNISTDataModule
from project_2.part_2.experiment_2 import perform_experiment_2
from project_2.part_2.models.kuba import MnistLargeModel


def _main():
    augmentation = v2.RandomAffine(degrees=10, scale=(0.6, 1.5), shear=20)

    def create_module(subset_size: int) -> pl.LightningDataModule:
        return AugmentedMNISTDataModule(
            transform=v2.Compose([v2.ToTensor()]),
            augmentation=augmentation,
            subset_size=subset_size,
            num_augmentations=10,
        )

    def create_model() -> pl.LightningModule:
        return MnistLargeModel(num_classes=10)

    perform_experiment_2(create_module, create_model, epochs=4)


if __name__ == '__main__':
    _main()
