import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CSVLogger

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models import ModelBase, saved_models_dir


class MagdaCifarLarge(ModelBase):
    def __init__(self, num_classes):
        feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 3x32x32 -> 32x32x32
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x32 -> 32x16x16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 32x16x16 -> 64x16x16
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x16x16 -> 64x8x8

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 64x8x8 -> 128x8x8
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x8x8 -> 128x4x4

            nn.Flatten(start_dim=1)  # Flatten 128x4x4 to 128*4*4

        )
        num_features = 128 * 4 * 4
        classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        super().__init__(feature_extractor, classifier, num_classes)


def get_model(trainer: pl.Trainer, data_module: pl.LightningDataModule) -> pl.LightningModule:
    model_path = saved_models_dir / 'magda_cifar_big.pt'
    try:
        model: pl.LightningModule = torch.load(model_path)
        print("Loaded model from disk")
        return model
    except FileNotFoundError:
        model = MagdaCifarLarge(num_classes=10)
        trainer.fit(model, data_module)
        print("Saving model to disk")
        torch.save(model, model_path)
        return model


def _main():
    torch.manual_seed(42)
    dm = CIFAR10DataModule()

    logger = CSVLogger("logs", name="magda_cifar_big")
    trainer = pl.Trainer(max_epochs=11, fast_dev_run=False, logger=logger)
    model = get_model(trainer, dm)

    trainer.test(model, dm)

    cm = model.confusion_matrix
    fig, ax = plt.subplots()
    sn.heatmap(cm.compute(), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix on CIFAR dataset, Archi 1 - Magda")
    fig.show()


if __name__ == '__main__':
    _main()
