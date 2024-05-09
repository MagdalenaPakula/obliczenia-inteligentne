import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassConfusionMatrix


def train_model(model, train_loader, epochs, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        running_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data = data
            labels = labels.long()

            # calculate loss
            optimizer.zero_grad()
            outputs = model(data.float())
            loss = criterion(outputs, labels)

            # update model parameters
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}")


def evaluate_model(model: nn.Module, test_loader: DataLoader, dataset_name: str):
    model.eval()

    y_pred = []
    y_true = []

    with torch.inference_mode():
        for data, labels in test_loader:
            data = data
            y_true.extend(labels.numpy())
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, dim=1)
            y_pred.extend(predicted.numpy())

    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    total = len(y_pred)
    correct = torch.eq(y_pred, y_true).sum()

    fig, ax = plt.subplots()
    ax.set_title(f"{dataset_name} — confusion matrix")
    cm = MulticlassConfusionMatrix(len(np.unique(y_true.numpy())))
    cm.update(y_pred, y_true)
    sn.heatmap(cm.normalized('true'), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.show()

    print(f'Dokładność na zbiorze testowym: {correct / total * 100:.2f}%')


def perform_experiment(dataset: dict[str, str | Dataset], model: nn.Module, epochs=50):
    name = dataset['name']
    train_dataset = dataset['train_dataset']
    test_dataset = dataset['test_dataset']

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_model(model, train_loader, epochs=epochs)
    evaluate_model(model, test_loader, name)
