import torch

from project_2.part_1.MLP import MLP
from project_2.part_1.experiments import perform_experiment
from project_2.part_1.features.two_dimentional.tsne import get_mnist_tsne

if __name__ == '__main__':
    dataset = get_mnist_tsne()

    torch.manual_seed(42)
    input_size = 2
    hidden_layer_size = 4
    output_size = 10

    model = MLP(input_size, hidden_layer_size, output_size)

    perform_experiment(dataset, model, epochs=150, learning_rate=0.05)
