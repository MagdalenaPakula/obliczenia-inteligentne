import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.data.other_datasets import load_other_datasets, CustomDataset


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_relu=True):
        super(MLP, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True) if use_relu else None  # Optional ReLU activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through layers
        x = self.fc1(x)
        if self.relu:
            x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, learning_rate=0.01, num_epochs=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        # Train phase
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Optional evaluation
        if test_loader is not None:
            with torch.no_grad():
                correct = 0
                total = 0
                model.eval()
                for data, target in test_loader:
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                accuracy = 100 * correct / total
                print(f"Test Accuracy: {accuracy:.2f}%")


