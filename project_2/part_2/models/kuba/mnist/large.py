import pytorch_lightning as pl
import torch
import torch.nn as nn

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models import ModelBase
from project_2.part_2.models.util import get_model, perform_experiment_1


class MnistLargeModel(ModelBase):
    def __init__(self, num_classes):
        feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),  # 6 * 28 * 28
            nn.Sigmoid(),
            nn.MaxPool2d(2),  # 6 * 14 * 14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 16 * 10 * 10
            nn.Sigmoid(),
            nn.MaxPool2d(2),  # 16 * 5 * 5
            nn.Flatten(start_dim=1),
        )
        num_features = 16 * 5 * 5
        classifier = nn.Sequential(
            nn.Linear(num_features, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )
        super().__init__(feature_extractor, classifier, num_classes)


def _main():
    torch.manual_seed(42)
    dm = MNISTDataModule()

    def factory():
        return MnistLargeModel(num_classes=10)

    trainer = pl.Trainer(max_epochs=50, fast_dev_run=False)
    model = get_model('kuba_mnist_big.pt', trainer, dm, factory)

    perform_experiment_1(model, 'MNIST Kuba-L', trainer, dm)


if __name__ == '__main__':
    _main()
