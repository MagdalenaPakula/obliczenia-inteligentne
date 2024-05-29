import pytorch_lightning as pl
from torchvision.transforms import v2

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.data.augmented import AugmentedCIFAR10DataModule
from project_2.part_2.experiment_2 import perform_experiment_2
from project_2.part_2.models.magda.cifar.small import MagdaCifarSmall


def _main():
    augmentation = v2.Identity()

    def create_module(subset_size: int) -> pl.LightningDataModule:
        return CIFAR10DataModule(
            transform=v2.Compose([
                v2.ToTensor(),
                augmentation]),
        )

    def create_model() -> pl.LightningModule:
        return MagdaCifarSmall(num_classes=10)

    perform_experiment_2(create_module, create_model, epochs=4)


if __name__ == '__main__':
    _main()
