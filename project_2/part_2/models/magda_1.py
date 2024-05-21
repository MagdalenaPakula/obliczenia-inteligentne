import os
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import argmax
from torchmetrics import Accuracy
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix

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
        self.confusion_matrix = None

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


def plot_loss(loss_dict) -> None:
    if torch.cuda.is_available():
        losses = [loss_dict['loss'][i].cpu() for i in range(len(loss_dict['loss']))]
        val_losses = [loss_dict['val_loss'][i].cpu() for i in range(len(loss_dict['val_loss']))]
    else:
        losses = loss_dict['loss']
        val_losses = loss_dict['val_loss']
    plt.figure('loss')
    plt.plot(losses, label='loss', c='black')
    plt.plot(val_losses, label='val_loss', c='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_accuracy(acc_dict) -> None:
    # Get accuracy values
    if torch.cuda.is_available():
        accuracy = [acc_dict['accuracy'][i].cpu() for i in range(len(acc_dict['accuracy']))]
        val_accuracy = [acc_dict['val_accuracy'][i].cpu() for i in range(len(acc_dict['val_accuracy']))]
    else:
        accuracy = acc_dict['accuracy']
        val_accuracy = acc_dict['val_accuracy']
    plt.figure('accuracy')
    plt.plot(accuracy, label='accuracy', c='black')
    plt.plot(val_accuracy, label='val_accuracy', c='red')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    tracker = MetricTrackerCallback()

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0.001,
        mode='min',
    )

    data_module = MNISTDataModule()
    model = Magda1MNIST()

    dirpath = Path.cwd()
    # Remove previous best model (if exists)
    if os.path.exists('best_model.ckpt'):
        os.remove('best_model.ckpt')
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="best_model",
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=10, enable_model_summary=True, callbacks=[tracker, early_stopping_callback, model_checkpoint_callback])

    # Trening
    trainer.fit(model, data_module)

    plot_loss(tracker.losses)
    plot_accuracy(tracker.acc)

    result = trainer.test(model, data_module, verbose=False, ckpt_path="best_model.ckpt")
    print(f"Accuracy in test data: {result[0]['test_accuracy']}")
    print(f"Loss in test data: {result[0]['test_loss']}")

    # Access confusion matrix from the best model
    confusion_matrix = model.confusion_matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.show()
