import os
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import argmax
from torchmetrics import Accuracy
import seaborn as sns

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models.magda.mnist.magda_MNIST_1 import MetricTrackerCallback, plot_accuracy


class Magda1CIFAR(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(32 * 8 * 8, 64)  # Smaller fully connected layer
        self.fc2 = nn.Linear(64, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_predictions = []
        self.test_targets = []
        self.confusion_matrix = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        y_pred = argmax(logits, dim=-1)
        acc = self.accuracy(y_pred, y)
        self.log('accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        y_pred = argmax(logits, dim=-1)
        acc = self.accuracy(y_pred, y)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        y_pred = argmax(logits, dim=-1)
        acc = self.accuracy(y_pred, y)
        self.log('test_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_pred.extend(y_pred.cpu().numpy())
        # Calculate and store confusion matrix
        self.test_predictions.extend(y_pred.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self.forward(x)
        y_pred = argmax(logits, dim=-1)
        self.pred_pred.extend(y_pred.cpu().numpy())
        return y_pred

    def test_epoch_end(self, outputs):
        # Compute confusion matrix using all predictions and targets
        self.confusion_matrix = confusion_matrix(self.test_targets, self.test_predictions)


if __name__ == '__main__':
    tracker = MetricTrackerCallback()
    data_module = CIFAR10DataModule()
    model = Magda1CIFAR()

    dirpath = Path.cwd()
    # Remove previous best model (if exists)
    if os.path.exists('best_model3.ckpt'):
        os.remove('best_model3.ckpt')
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="best_model3",
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=2, enable_model_summary=True, callbacks=[tracker, model_checkpoint_callback])

    trainer.fit(model, data_module)

    plot_accuracy(tracker.acc)

    result = trainer.test(model, data_module, verbose=False, ckpt_path='best_model3.ckpt')
    print(f"Accuracy in test data: {result[0]['test_accuracy']}")
    print(f"Loss in test data: {result[0]['test_loss']}")

    # Access confusion matrix from the best model
    confusion_matrix = model.confusion_matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.show()
