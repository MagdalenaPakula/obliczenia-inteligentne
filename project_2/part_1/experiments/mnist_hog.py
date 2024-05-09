import torch

from project_2.part_1.MLP import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.experiments import perform_experiment
from project_2.part_1.features.reduced_dimention.hog import hog_feature_extraction

if __name__ == '__main__':
    dataset = load_dataset_MNIST(transform=hog_feature_extraction)

    for hidden in [12]:
        torch.manual_seed(42)
        input_size = dataset['train_dataset'][0][0].size(0)
        hidden_layer_size = hidden
        output_size = 10

        model = MLP(input_size, hidden_layer_size, output_size)

        perform_experiment(dataset, model, epochs=100, learning_rate=0.1)
