import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassConfusionMatrix
import seaborn as sn
from project_2.part_1.MLP import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.experiments import train_model
from project_2.part_1.features.two_dimentional.pca import pca_features


def evaluate_model_pca(model: MLP, test_loader: DataLoader, dataset_name: str):
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, labels in test_loader:
            pca_features_data = pca_features(data.numpy())
            outputs = model(torch.tensor(pca_features_data, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

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
    plt.show()

    print(f'Accuracy on the test set: {correct.item() / total * 100:.2f}%')


def perform_experiment_pca(dataset: dict[str, str | Dataset], model: MLP, epochs=50, learning_rate=0.01):
    name = dataset['name']
    train_images = dataset['train_dataset'].data
    train_targets = dataset['train_dataset'].targets

    # Perform PCA on train images
    pca_train_features = pca_features(train_images)

    # Create train loader with PCA features
    train_loader = DataLoader(list(zip(pca_train_features, train_targets)), batch_size=64, shuffle=True)

    # Train model
    train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate)

    # Evaluate the trained model on the test set
    test_images = dataset['test_dataset'].data
    test_targets = dataset['test_dataset'].targets
    pca_test_features = pca_features(test_images)
    test_loader = DataLoader(list(zip(pca_test_features, test_targets)), batch_size=64, shuffle=False)
    evaluate_model_pca(model, test_loader, name)


if __name__ == "__main__":
    # Load MNIST datasets
    datasets_mnist = load_dataset_MNIST()

    # Create MLP model
    input_dim = 2  # Two principal components
    hidden_dim = 4
    output_dim = 10  # 10 classes (0-9)
    model = MLP(input_dim, hidden_dim, output_dim)

    # Perform experiment
    perform_experiment_pca(datasets_mnist, model, epochs=50, learning_rate=0.001)
