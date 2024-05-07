import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.data.other_datasets import load_other_datasets, CustomDataset
import torch.optim as optim


# Definicja klasy MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_relu=True):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True) if use_relu else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Funkcja trenująca model
def train_model(model, train_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01) # learning rate here

    model.to(device)

    for epoch in range(epochs):
        print(f"Epoka {epoch + 1}/{epochs}")
        running_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(data.float())
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f"[Step {i + 1}/{len(train_loader)}] Loss: {running_loss / 100}")
                running_loss = 0.0



# Funkcja ewaluująca model
def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Dokładność na zbiorze testowym: {correct / total * 100}%')


if __name__ == "__main__":
    input_dim = [4, 13, 30]  # Input dimensions for: Iris, Wine & Breast Cancer
    hidden_dim = 64
    epochs = 50

    datasets = load_other_datasets()

    for dataset in datasets:
        train_dataset = dataset['train_dataset']
        test_dataset = dataset['test_dataset']

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        target_labels = []
        for _, label in train_dataset:
            target_labels.append(label.item())

        num_classes = len(set(target_labels))

        model = MLP(input_dim[datasets.index(dataset)], hidden_dim, num_classes)

        train_model(model, train_loader, 'cpu', epochs)
        evaluate_model(model, test_loader, 'cpu')

