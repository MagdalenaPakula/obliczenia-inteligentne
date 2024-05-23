import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models import ModelBase, saved_models_dir


class MagdaMnistLarge(ModelBase):
    def __init__(self, num_classes):
        feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 5, 1, 2),   # 8 * 28 * 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8 * 14 * 14
            nn.Conv2d(8, 16, 5, 1, 2),  # 16 * 14 * 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 * 7 * 7
            nn.Conv2d(16, 32, 5, 1, 2),  # 32 * 7 * 7
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 * 3 * 3
            nn.Flatten(start_dim=1),
        )
        num_features = 32 * 3 * 3  # 288
        classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
        super().__init__(feature_extractor, classifier, num_classes)


def get_model(trainer: pl.Trainer, data_module: pl.LightningDataModule) -> pl.LightningModule:
    model_path = saved_models_dir / 'magda_mnist_big.pt'
    try:
        model: pl.LightningModule = torch.load(model_path)
        print("Loaded model from disk")
        return model
    except FileNotFoundError:
        model = MagdaMnistLarge(num_classes=10)
        trainer.fit(model, data_module)
        print("Saving model to disk")
        torch.save(model, model_path)
        return model


def _main():
    torch.manual_seed(42)
    dm = MNISTDataModule()

    trainer = pl.Trainer(max_epochs=11, fast_dev_run=False)
    model = get_model(trainer, dm)

    trainer.test(model, dm)

    cm = model.confusion_matrix
    fig, ax = plt.subplots()
    sn.heatmap(cm.compute(), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix on MNIST dataset, Archi 1 - Magda")
    fig.show()


if __name__ == '__main__':
    _main()



