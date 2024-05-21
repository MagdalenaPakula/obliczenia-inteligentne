import numpy as np
import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim, argmax
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy

from project_2.part_2.data import MNISTDataModule


# 1D vector, num of class = 10 (liczby 0-9)
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
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_pred = []  # collect predictions
        self.pred_pred = []  # collect predictions
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('loss', loss)
        # Track accuracy
        y_pred = argmax(logits, dim=-1)
        acc = self.accuracy(y_pred, y)
        self.log('accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)
        # Track accuracy
        y_pred = argmax(logits, dim=-1)
        acc = self.accuracy(y_pred, y)
        self.log('val_accuracy', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate model
        logits = self.forward(x)
        # Track loss
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        # Track accuracy
        y_pred = argmax(logits, dim=-1)  # find label with highest probability
        acc = self.accuracy(y_pred, y)
        self.log('test_accuracy', acc)
        # Collect predictions
        self.test_pred.extend(y_pred.cpu().numpy())
        # Update confusion matrix
        self.confusion_matrix.update(y_pred, y)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self.forward(x)
        y_pred = argmax(logits, dim=-1)
        # Collect predictions
        self.pred_pred.extend(y_pred.cpu().numpy())
        return y_pred


class MetricTrackerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {
            'loss': [],
            'val_loss': []
        }
        self.acc = {
            'accuracy': [],
            'val_accuracy': []
        }

    def on_train_epoch_end(self, trainer, module):
        metrics = trainer.logged_metrics
        self.losses['loss'].append(metrics['loss'])
        self.acc['accuracy'].append(metrics['accuracy'])

    def on_validation_epoch_end(self, trainer, module):
        metrics = trainer.logged_metrics
        self.losses['val_loss'].append(metrics['val_loss'])
        self.acc['val_accuracy'].append(metrics['val_accuracy'])


if __name__ == '__main__':
    torch.manual_seed(42)

    data_module = MNISTDataModule()

    model = Magda1MNIST()

    # Inicjalizacja trainera
    trainer = pl.Trainer(max_epochs=2)

    # Trening
    trainer.fit(model, data_module)

    # Testowanie
    trainer.test(model=model, datamodule=data_module, )
