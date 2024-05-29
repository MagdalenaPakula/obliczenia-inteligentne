import pytorch_lightning as pl
from torchvision.transforms import v2

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.data.augmented import AugmentedMNISTDataModule
from project_2.part_2.experiment_2 import perform_experiment_2
from project_2.part_2.models.kuba.mnist.large import MnistLargeModel


def _main():
    augmentation = v2.ElasticTransform(alpha=50.0, sigma=5.0)

    def create_module(subset_size: int) -> pl.LightningDataModule:
        return MNISTDataModule(
            transform=v2.Compose([
                v2.ToTensor(),
                augmentation]),
        )

    def create_model() -> pl.LightningModule:
        return MnistLargeModel(num_classes=10)

    perform_experiment_2(create_module, create_model, epochs=4)


if __name__ == '__main__':
    _main()
