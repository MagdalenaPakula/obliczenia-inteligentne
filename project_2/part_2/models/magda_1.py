import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy

from project_2.part_2.data import MNISTDataModule


# num of class = 10 (liczby 0-9)
class Magda1MNIST(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 3 * 3, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.accuracy(logits, y)
        self.log_dict({
            'train_loss': loss.item(),
            'train_acc': self.accuracy.compute(),
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.accuracy(logits, y)
        self.log_dict({
            'test_loss': loss.item(),
            'test_acc': self.accuracy.compute(),
        })
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == '__main__':
    torch.manual_seed(42)

    data_module = MNISTDataModule()

    model = Magda1MNIST()

    # Inicjalizacja trainera
    trainer = pl.Trainer(max_epochs=100)

    # Trening
    trainer.fit(model, data_module)

    # Testowanie
    trainer.test(model=model, datamodule=data_module)
