import torch
from torch import nn

from project_2.part_1.MLP import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.experiments import perform_experiment, train_model, evaluate_model
from project_2.part_1.features.reduced_dimention.lbp import lbp_image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from project_1.visualization import plot_decision_boundary
from project_2.part_1.MLP import MLP
from project_2.part_1.experiments import train_model, evaluate_model
from project_2.part_1.features.two_dimentional.tsne import get_mnist_tsne

if __name__ == "__main__":
    dataset = load_dataset_MNIST(transform=lambda x: lbp_image(x, radius=1, n_points=8))

    for hidden in [24]: # use 12-32
        torch.manual_seed(42)
        input_size = dataset['train_dataset'][0][0].size(0)
        hidden_layer_size = hidden
        output_size = 10

        model = MLP(input_size, hidden_layer_size, output_size)

        perform_experiment(dataset, model, epochs=5, learning_rate=0.01)