import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix


class _ModelBase(pl.LightningModule):
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
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
