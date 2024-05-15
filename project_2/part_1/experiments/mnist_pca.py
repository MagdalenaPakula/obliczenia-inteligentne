import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassConfusionMatrix
import seaborn as sn
from project_1.visualization import plot_decision_boundary, plot_voronoi_diagram
from project_1.visualization import plot_voronoi_diagram, plot_decision_boundary
from project_2.part_1.MLP import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST, CustomMNISTDataset
from project_2.part_1.experiments import train_model, evaluate_model
from project_2.part_1.features.two_dimentional.pca import pca_features


def perform_experiment(dataset: dict[str, str | Dataset], model: nn.Module, epochs=50, learning_rate=0.01):
    name = dataset['name']
    train_dataset = dataset['train_dataset']
    test_dataset = dataset['test_dataset']

    # PCA transformation
    pca_train_data = pca_features(train_dataset.data)
    pca_test_data = pca_features(test_dataset.data)

    # Create loaders for PCA-transformed data
    pca_train_loader = DataLoader(CustomMNISTDataset(pca_train_data, train_dataset.targets),
                                  batch_size=64, shuffle=True)
    pca_test_loader = DataLoader(CustomMNISTDataset(pca_test_data, test_dataset.targets),
                                 batch_size=64, shuffle=False)

    # Train model
    train_model(model, pca_train_loader, epochs=epochs, learning_rate=learning_rate)

    # Evaluate model
    evaluate_model(model, pca_test_loader, name)

    # Generate Voronoi diagram
    voronoi_diagram(pca_train_data, model)

    # Generate Decision boundary
    decision_boundary(pca_train_data, train_dataset.targets, model)


# Voronoi diagram function
def voronoi_diagram(data, model):
    predicted_labels = model(torch.tensor(data).float()).argmax(dim=1).detach().numpy()
    plot_voronoi_diagram(data, predicted_labels)


# Decision boundary function
def decision_boundary(data, labels, model):
    def classifier(x):
        with torch.no_grad():
            return model(torch.tensor(x).float()).argmax(dim=1).detach().numpy()

    plot_decision_boundary(classifier, data, labels, title="Decision Boundary")


if __name__ == "__main__":
    # Load MNIST dataset
    mnist_dataset = load_dataset_MNIST()

    # Define MLP model
    model = MLP(input_dim=2, hidden_dim=256, output_dim=10)

    # Perform experiment
    perform_experiment(mnist_dataset, model)
