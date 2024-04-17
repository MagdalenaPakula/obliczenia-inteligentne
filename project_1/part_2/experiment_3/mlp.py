import warnings
from typing import Dict, Tuple, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from project_1.data import load_generated_datasets
from project_1.visualization import plot_decision_boundary

type Classifier = MLPClassifier

warnings.filterwarnings("ignore")


def experiment_3(dataset, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=0.2,
                                                        test_size=0.2,
                                                        random_state=42)

    layer_sizes = range(1, 10)
    classifiers: List[Classifier] = []
    train_scores = []
    test_scores = []
    for num_neurons in layer_sizes:
        classifier = MLPClassifier(hidden_layer_sizes=(num_neurons,),
                                   activation='relu',
                                   max_iter=100_000,
                                   tol=0,
                                   n_iter_no_change=100_000,
                                   solver='sgd',
                                   random_state=42)
        print(f'Training network with {num_neurons} neurons on dataset {dataset_name}... ', end='')
        classifier.fit(X_train, y_train)
        print('Done')

        train_scores.append(classifier.score(X_train, y_train))
        test_scores.append(classifier.score(X_test, y_test))
        classifiers.append(classifier)

    plt.plot(layer_sizes, train_scores, label='treningowy')
    plt.plot(layer_sizes, test_scores, label='testowy')
    plt.grid(linestyle='--', axis='x')
    plt.xlabel('Hidden layer size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.title(f'MLP na zbiorze danych {dataset_name}')
    plt.show()
    plt.close()

    data: Dict[str, Tuple[ndarray, ndarray, List[float]]] = {'treningowy': (X_train, y_train, train_scores),
                                                             'testowy': (X_test, y_test, test_scores)}

    min_classifier = classifiers[0]
    max_classifier = classifiers[-1]
    for set_choice, (features, labels, scores) in data.items():
        best_classifier = classifiers[np.argmax(scores)]

        selected_classifiers: Dict[str, Classifier] = {'Min': min_classifier,
                                                       'Best': best_classifier,
                                                       'Max': max_classifier}
        for classifier_choice, classifier in selected_classifiers.items():
            plot_decision_boundary(lambda x: classifier.predict(x), features, labels,
                                   title=f'{classifier_choice} MLP na zbiorze {dataset_name} - zbiór {set_choice}')
            plt.show()
            plt.close()

            cm = confusion_matrix(y_train, classifier.predict(X_train))
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title(f"Macierz pomyłek dla {classifier_choice} MLP na {dataset_name} - zbiór {set_choice} ")
            plt.xlabel("Przewidziane etykiety")
            plt.ylabel("Prawdziwe etykiety")
            plt.show()


if __name__ == '__main__':
    for dataset, name in load_generated_datasets()[4:]:
        experiment_3(dataset, name)
