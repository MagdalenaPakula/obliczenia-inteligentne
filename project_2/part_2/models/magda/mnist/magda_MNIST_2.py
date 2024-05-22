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

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models.magda.mnist.magda_MNIST_1 import MetricTrackerCallback, plot_loss, plot_accuracy


# gives 2 FEATURES + decision boundary
class Magda2MNIST(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # Reduced filters from 32 to 16
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),  # Reduced filters from 32 to 16
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(8 * 7 * 7, 2),  # 2 features
            nn.ReLU(),
            nn.Linear(2, 10)  # 10 classes
        )

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_pred = []
        self.pred_pred = []
        self.confusion_matrix = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output

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
        self.confusion_matrix = confusion_matrix(y, y_pred)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self.forward(x)
        y_pred = argmax(logits, dim=-1)
        self.pred_pred.extend(y_pred.cpu().numpy())
        return y_pred

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        features = self.fc[0](x)  # Extract features after the first linear layer
        return features


if __name__ == '__main__':

    tracker = MetricTrackerCallback()
    data_module = MNISTDataModule()
    model = Magda2MNIST()

    dirpath = Path.cwd()
    # Remove previous best model (if exists)
    if os.path.exists('best_model2.ckpt'):
        os.remove('best_model2.ckpt')
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="best_model2",
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=6, enable_model_summary=True, callbacks=[tracker, model_checkpoint_callback])

    trainer.fit(model, data_module)

    plot_accuracy(tracker.acc)

    result = trainer.test(model, data_module, verbose=False, ckpt_path='best_model2.ckpt')
    print(f"Accuracy in test data: {result[0]['test_accuracy']}")
    print(f"Loss in test data: {result[0]['test_loss']}")

    # Access confusion matrix from the best model
    confusion_matrix = model.confusion_matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.show()

    # Add decision boundary plot generation