import pytorch_lightning as pl
import torch
from torch import nn

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models import ModelBase
from project_2.part_2.models.util import get_model, perform_experiment_1


def _conv_module(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class _CIFARLargeFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.step_1 = nn.Sequential(
            _conv_module(3, 64),
            _conv_module(64, 128),
            nn.MaxPool2d(kernel_size=2),
        )
        self.residual_1 = nn.Sequential(
            _conv_module(128, 128),
            _conv_module(128, 128),
        )
        self.step_2 = nn.Sequential(
            _conv_module(128, 64),
            nn.MaxPool2d(2),
        )
        self.residual_2 = nn.Sequential(
            _conv_module(64, 64),
            _conv_module(64, 64),
        )
        self.step_3 = nn.Sequential(
            _conv_module(64, 16),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: 3 * 32 * 32
        x = self.step_1(x)  # 128 * 16 * 16
        x = self.residual_1(x) + x
        x = self.step_2(x)  # 64 * 8 * 8
        x = self.residual_2(x) + x
        x = self.step_3(x)  # 16 * 4 * 4

        flat = x.flatten(start_dim=1)
        return flat


class CifarLargeModel(ModelBase):
    def __init__(self):
        feature_extractor = _CIFARLargeFeatureExtractor()
        classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        super().__init__(feature_extractor, classifier, num_classes=10)


def _main():
    dm = CIFAR10DataModule()
    trainer = pl.Trainer(max_epochs=10, fast_dev_run=False)

    def factory():
        return CifarLargeModel()

    model = get_model('kuba_cifar_large.pt', trainer, dm, factory)
    perform_experiment_1(model, 'CIFAR Kuba-L', trainer, dm)


if __name__ == '__main__':
    _main()
