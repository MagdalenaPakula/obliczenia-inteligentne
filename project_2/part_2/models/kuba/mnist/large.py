import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models import _ModelBase, saved_models_dir


class MnistLargeModel(_ModelBase):
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


def get_model(trainer: pl.Trainer, data_module: pl.LightningDataModule) -> pl.LightningModule:
    model_path = saved_models_dir / 'kuba_mnist_big.pt'
    try:
        model: pl.LightningModule = torch.load(model_path)
        print("Loaded model from disk")
        return model
    except FileNotFoundError:
        model = MnistLargeModel(num_classes=10)
        trainer.fit(model, data_module)
        torch.save(model, model_path)
        return model


def _main():
    torch.manual_seed(42)
    dm = MNISTDataModule()

    trainer = pl.Trainer(max_epochs=50, fast_dev_run=False)
    model = get_model(trainer, dm)

    trainer.test(model, dm)

    cm = model.confusion_matrix
    fig, ax = plt.subplots()
    sn.heatmap(cm.compute(), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix on MNIST dataset, Kuba-L model")
    fig.show()


if __name__ == '__main__':
    _main()
