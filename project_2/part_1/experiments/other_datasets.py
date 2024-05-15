import numpy as np
import torch

from project_2.part_1.MLP import MLP
from project_2.part_1.data.other_datasets import load_other_datasets
from project_2.part_1.experiments import perform_experiment

if __name__ == '__main__':

    # found by trial and error
    hidden_layer_sizes = {
        "Iris Dataset": 7,
        "Wine Dataset": 2,
        "Breast Cancer Wisconsin Dataset": 3
    }

    torch.manual_seed(42)
    for dataset in load_other_datasets():
        print(f"Testing {dataset['name']}")

        input_size: int = dataset['train_dataset'][0][0].size
        output_size: int = len(np.unique(dataset['train_dataset'].target))
        hidden_size = hidden_layer_sizes[dataset['name']]

        model = MLP(input_size, hidden_size, output_size)

        perform_experiment(dataset, model, epochs=200, learning_rate=0.1)
