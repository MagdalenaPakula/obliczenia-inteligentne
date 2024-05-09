import torch

from project_2.part_1.MLP import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.experiments import perform_experiment

if __name__ == '__main__':
    dataset = load_dataset_MNIST(transform=torch.flatten)

    torch.manual_seed(42)
    input_size = 28 * 28
    hidden_layer_size = 20
    output_size = 10

    model = MLP(input_size, hidden_layer_size, output_size)

    perform_experiment(dataset, model, epochs=100)
