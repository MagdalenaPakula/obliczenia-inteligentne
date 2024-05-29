import pytorch_lightning as pl
from torchvision.transforms import v2

from project_2.part_2.data.augmented import AugmentedCIFAR10DataModule
from project_2.part_2.experiment_2 import perform_experiment_2
from project_2.part_2.models.magda.cifar.small import MagdaCifarSmall


def _main():
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1, 1.2), shear=10),
        v2.Resize((36, 36)),
        v2.RandomCrop((32, 32)),
        v2.ColorJitter(brightness=0.2, hue=0.05, saturation=0.1)
    ])

    def create_module(subset_size: int) -> pl.LightningDataModule:
        return AugmentedCIFAR10DataModule(
            transform=v2.Compose([v2.ToTensor()]),
            augmentation=augmentation,
            subset_size=subset_size,
            num_augmentations=10,
        )

    def create_model() -> pl.LightningModule:
        return MagdaCifarSmall(num_classes=10)

    perform_experiment_2(create_module, create_model, epochs=4)


if __name__ == '__main__':
    _main()
