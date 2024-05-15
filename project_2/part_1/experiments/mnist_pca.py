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
from project_2.part_1.experiments import train_model, evaluate_model, perform_experiment
# from project_2.part_1.features.two_dimentional.pca import pca_features


# def perform_experiment(dataset, model, epochs=100, learning_rate=0.001):
#     name = dataset['name']
#     train_dataset = dataset['train_dataset']
#     test_dataset = dataset['test_dataset']
#
#     pca = PCA(n_components=2)
#     pca.fit(train_dataset.data.numpy().reshape(len(train_dataset), -1))
#     train_data_pca = pca.transform(train_dataset.data.numpy().reshape(len(train_dataset), -1))
#     test_data_pca = pca.transform(test_dataset.data.numpy().reshape(len(test_dataset), -1))
#
#     train_dataset_pca = CustomMNISTDataset(data=train_data_pca, target=train_dataset.targets)
#     test_dataset_pca = CustomMNISTDataset(data=test_data_pca, target=test_dataset.targets)
#
#     train_loader_pca = DataLoader(train_dataset_pca, batch_size=64, shuffle=True)
#     test_loader_pca = DataLoader(test_dataset_pca, batch_size=64, shuffle=False)
#
#     train_model(model, train_loader_pca, epochs=epochs, learning_rate=learning_rate)
#     evaluate_model(model, test_loader_pca, name)
#
#
# if __name__ == "__main__":
#     input_dim = 2  # 2 for PCA-transformed data
#     hidden_dim = 128
#     output_dim = 10  # Number of classes
#     model = MLP(input_dim, hidden_dim, output_dim)
#
#     mnist_dataset = load_dataset_MNIST()
#
#     perform_experiment(mnist_dataset, model)
def pca_features(data, n_components=2):
    if len(data.shape) > 2:
        flattened_data = data.reshape(-1, data.shape[1] * data.shape[2])
    else:
        flattened_data = data

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flattened_data)

    # Create a PCA object
    pca = PCA(n_components=min(n_components, min(data.shape[1] * data.shape[2], data.shape[0])))

    # Fit the PCA to the data
    pca.fit(scaled_data)

    # Transform the data using the fitted PCA
    transformed_data = pca.transform(scaled_data)

    return transformed_data

if __name__ == "__main__":
    dataset = load_dataset_MNIST(transform=lambda x: pca_features(x, n_components=2))

    for hidden in [12]: # use 12-32
        torch.manual_seed(42)
        input_size = dataset['train_dataset'][0][0].size(0)
        hidden_layer_size = hidden
        output_size = 10

        model = MLP(input_size, hidden_layer_size, output_size)

        perform_experiment(dataset, model, epochs=5, learning_rate=0.01)


