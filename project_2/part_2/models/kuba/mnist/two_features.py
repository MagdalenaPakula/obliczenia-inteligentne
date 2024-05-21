import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy

from project_2.part_2.data import MNISTDataModule


class Mnist2FeatureModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.extract_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),  # 10 * 28 * 28
            nn.MaxPool2d(2),  # 6 * 12 * 12
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 20 * 10 * 10
            nn.MaxPool2d(2),  # 16 * 5 * 5
            nn.Flatten(start_dim=1),
            nn.Sequential(nn.Linear(16 * 5 * 5, 2), nn.ReLU())
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes)
        )
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.fully_connected(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.accuracy(logits, y)
        self.log_dict({
            'train_loss': loss.item(),
            'train_acc': self.accuracy
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.inference_mode():
            x, y = batch
            logits = self(x)
            loss = self.loss(logits, y)
            self.val_accuracy(logits, y)
            self.log_dict({
                'val_loss': loss.item(),
                'val_acc': self.val_accuracy
            }, on_step=False, on_epoch=True, prog_bar=True)
        self.train()
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.accuracy.update(y_hat, y)
        self.confusion_matrix.update(y_hat, y)

        self.log_dict({
            'test_loss': loss.item(),
            'test_acc': self.accuracy
        })


if __name__ == '__main__':
    torch.manual_seed(42)

    dm = MNISTDataModule()

    model = Mnist2FeatureModel(num_classes=10)
    trainer = pl.Trainer(max_epochs=100, fast_dev_run=True)
    trainer.fit(model, dm)
    test_data = trainer.test(model, dm)

    cm = model.confusion_matrix

    fig, ax = plt.subplots()
    sn.heatmap(cm.compute(), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.show()
