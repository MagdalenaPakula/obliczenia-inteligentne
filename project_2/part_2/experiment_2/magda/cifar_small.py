import pytorch_lightning as pl
from torchvision.transforms import v2

from project_2.part_2.data.augmented import AugmentedCIFAR10DataModule
from project_2.part_2.experiment_2 import perform_experiment_2
from project_2.part_2.models.magda.cifar.new_small import MagdaCifarSmall


def _main():
    augmentation = v2.Compose([
        v2.RandomRotation(degrees=30),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.GaussianBlur(kernel_size=3)
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
