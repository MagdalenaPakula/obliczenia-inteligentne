import pathlib
import warnings

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from project_1.data import load_generated_datasets
from project_1.visualization import plot_decision_boundary

warnings.filterwarnings("ignore")


def experiment_1(dataset: ndarray, dataset_name: str):
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    activations = ['identity', 'relu']

    for activation in activations:
        neurons = range(1, 11)
        train_scores = []
        test_scores = []
        classifiers = []
        for num_neurons in neurons:
            mlp = MLPClassifier(hidden_layer_sizes=(num_neurons,),
                                activation=activation,
                                max_iter=100_000,
                                tol=0,
                                n_iter_no_change=100_000,
                                solver='sgd',
                                random_state=42)
            print(f'Training MLP: {num_neurons} neurons, {activation} activation, dataset {dataset_name}... ', end='')
            mlp.fit(X_train, y_train)
            print('Done')
            classifiers.append(mlp)

            train_score = mlp.score(X_train, y_train)
            train_scores.append(train_score)
            test_score = mlp.score(X_test, y_test)
            test_scores.append(test_score)

        plt.plot(neurons, train_scores, label='train set')
        plt.plot(neurons, test_scores, label='test set')
        plt.grid(linestyle='--', axis='x')
        plt.xlabel('Hidden layer neurons')
        plt.ylabel('score')
        plt.legend(loc='best')
        plt.title(f'MLP scores with activation {activation} on dataset {dataset_name}')
        pathlib.Path(f'mlp/{dataset_name[:-4]}/{activation}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'mlp/{dataset_name[:-4]}/{activation}/scores.png')
        plt.show()
        plt.close()

        best_idx = np.argmax(
            [test_score + 1e-5 * train_score for (test_score, train_score) in zip(test_scores, train_scores)])
        best_classifier = classifiers[best_idx]
        print(f'{dataset_name}/{activation} best score for n={neurons[best_idx]}')

        plot_decision_boundary(lambda x: best_classifier.predict(x), dataset[:, :2], dataset[:, -1],
                               title=f'MLP with activation {activation}, dataset {dataset_name}')
        plt.savefig(f'mlp/{dataset_name[:-4]}/{activation}/boundary.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    for dataset, name in load_generated_datasets()[3:]:
        experiment_1(dataset, name)
