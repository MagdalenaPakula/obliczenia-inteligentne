import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CSVLogger

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models import ModelBase, saved_models_dir
from project_2.part_2.visualization import plot_decision_boundary


class MagdaMnistSmall(ModelBase):
    def __init__(self, num_classes):
        feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x28x28 -> 32x14x14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 32x14x14 -> 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x14x14 -> 64x7x7

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 64x7x7 -> 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x7x7 -> 128x3x3

            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=0),  # 128x3x3 -> 2x1x1
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # Flatten 1x1x2 to 2
        )
        num_features = 2
        classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
        super().__init__(feature_extractor, classifier, num_classes)


def get_model(trainer: pl.Trainer, data_module: pl.LightningDataModule) -> pl.LightningModule:
    model_path = saved_models_dir / 'magda_mnist_small.pt'
    try:
        model: MagdaMnistSmall = torch.load(model_path)
        print("Loaded model from disk")
        return model
    except FileNotFoundError:
        model = MagdaMnistSmall(num_classes=10)
        trainer.fit(model, data_module)
        print("Saving model to disk")
        torch.save(model, model_path)
        return model


def _main():
    torch.manual_seed(42)
    dm = MNISTDataModule()

    logger = CSVLogger("logs", name="magda_mnist_small")
    trainer = pl.Trainer(max_epochs=11, fast_dev_run=False, logger=logger)
    model = get_model(trainer, dm)

    trainer.test(model, dm)

    cm = model.confusion_matrix
    fig, ax = plt.subplots()
    sn.heatmap(cm.compute(), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix on MNIST dataset, Archi 2 - Magda")
    fig.show()

    plot_decision_boundary(model, dm.test_dataloader())
    plt.title("Decision boundary on MINST dataset, Archi 2 - Magda")
    plt.show()

    print(model)


if __name__ == '__main__':
    _main()