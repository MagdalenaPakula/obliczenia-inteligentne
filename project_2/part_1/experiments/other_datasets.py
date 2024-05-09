import numpy as np

from project_2.part_1.MLP import MLP
from project_2.part_1.data.other_datasets import load_other_datasets
from project_2.part_1.experiments import perform_experiment

if __name__ == '__main__':

    for dataset in load_other_datasets():
        print(f"Testing {dataset['name']}")

        input_size: int = dataset['train_dataset'][0][0].size
        output_size: int = len(np.unique(dataset['train_dataset'].target))

        model = MLP(input_size, 64, output_size)

        perform_experiment(dataset, model)
