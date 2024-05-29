import pytorch_lightning as pl
import torch
import torch.nn as nn

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models import ModelBase
from project_2.part_2.models.util import get_model, perform_experiment_1


class MnistSmolModel(ModelBase):
    def __init__(self, num_classes):
        feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),  # 10 * 28 * 28
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),  # 10 * 14 * 14
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=5),  # 5 * 10 * 10
            nn.ReLU(),
            nn.MaxPool2d(2),  # 5 * 5 * 5
            nn.Conv2d(in_channels=5, out_channels=2, kernel_size=3),  # 2 * 3 * 3
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(2 * 3 * 3, 2),
        )
        classifier = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes)
        )
        super().__init__(feature_extractor, classifier, num_classes)


def _main():
    torch.manual_seed(42)
    dm = MNISTDataModule()

    def factory():
        return MnistSmolModel(num_classes=10)

    trainer = pl.Trainer(max_epochs=30, fast_dev_run=False)
    model = get_model('kuba_mnist_smol.pt', trainer, dm, factory)

    perform_experiment_1(model, 'MNIST Kuba-S model', trainer, dm, show_decision_boundary=True)


if __name__ == '__main__':
    _main()
