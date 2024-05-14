import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassConfusionMatrix
import seaborn as sn
from project_2.part_1.MLP import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.experiments import train_model, evaluate_model
from project_2.part_1.features.two_dimentional.pca import pca_features


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
    evaluate_model(model, test_loader, name)


if __name__ == "__main__":
    # Load MNIST datasets
    datasets_mnist = load_dataset_MNIST()

    # Create MLP model
    input_dim = 2  # Two principal components
    hidden_dim = 4
    output_dim = 10  # 10 classes (0-9)
    model = MLP(input_dim, hidden_dim, output_dim)

    # Perform experiment
    perform_experiment_pca(datasets_mnist, model, epochs=50, learning_rate=0.01)
