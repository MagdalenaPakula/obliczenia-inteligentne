import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models import ModelBase, saved_models_dir
from project_2.part_2.visualization import plot_decision_boundary


class MnistSmolModel(ModelBase):
    def __init__(self, num_classes):
        feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),  # 10 * 28 * 28
            nn.Sigmoid(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),  # 6 * 12 * 12
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 20 * 10 * 10
            nn.Sigmoid(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),  # 16 * 5 * 5
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5),  # 2 * 1 * 1
            nn.Sigmoid(),
            nn.Flatten(start_dim=1),
        )
        classifier = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes)
        )
        super().__init__(feature_extractor, classifier, num_classes)


def get_model(trainer: pl.Trainer, data_module: pl.LightningDataModule) -> MnistSmolModel:
    model_path = saved_models_dir / 'kuba_mnist_smol.pt'
    try:
        model: MnistSmolModel = torch.load(model_path)
        print("Loaded model from disk")
        return model
    except FileNotFoundError:
        model = MnistSmolModel(num_classes=10)
        trainer.fit(model, data_module)
        print("Saving model to disk")
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
    ax.set_title("Confusion matrix on MNIST dataset, Kuba-S model")
    fig.show()

    plot_decision_boundary(model, dm.test_dataloader())
    plt.title("Decision boundary on MINST dataset, Kuba-S model")
    plt.show()

    print(model)


if __name__ == '__main__':
    _main()
