import pytorch_lightning as pl
from torch import nn

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models import ModelBase
from project_2.part_2.models.util import get_model, perform_experiment_1


class CIFARSmol(ModelBase):
    def __init__(self):
        feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 64 * 32 * 32
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 * 16 * 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),  # 8 * 8 * 8
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 2, kernel_size=4),  # 2 * 1 * 1
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        classifier = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        super().__init__(feature_extractor, classifier, num_classes=10)


def _main():
    dm = CIFAR10DataModule()
    trainer = pl.Trainer(max_epochs=50, fast_dev_run=False)

    def factory():
        return CIFARSmol()

    model = get_model('kuba_cifar_smol.pt', trainer, dm, factory)
    perform_experiment_1(model, 'CIFAR Kuba-S', trainer, dm)


if __name__ == '__main__':
    _main()
