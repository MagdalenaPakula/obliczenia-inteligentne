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

Classifier = MLPClassifier

warnings.filterwarnings("ignore")


def experiment_4(dataset, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    layer_sizes = range(1, 10)
    classifiers: List[MLPClassifier] = []
    train_scores = []
    test_scores = []

    for num_neurons in layer_sizes:
        classifier = MLPClassifier(hidden_layer_sizes=(num_neurons,),
                                   activation='relu',
                                   max_iter=100000,
                                   tol=0,
                                   n_iter_no_change=100000,
                                   solver='sgd',
                                   random_state=42)

        train_acc = []
        test_acc = []
        print(f'Training network with {num_neurons} neurons on dataset {dataset_name}...')
        for _ in range(1000):  # 100,000 epochs
            classifier.partial_fit(X_train, y_train, classes=np.unique(y_train))
            train_acc.append(classifier.score(X_train, y_train))
            test_acc.append(classifier.score(X_test, y_test))

        train_scores.append(train_acc)
        test_scores.append(test_acc)
        classifiers.append(classifier)

    # Plot accuracy changes on training and test sets for subsequent epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 1001), train_scores[0], label='Training')
    plt.plot(range(1, 1001), test_scores[0], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Changes on Training and Test Sets for MLP on Dataset {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    # Find best classifier based on test set
    best_classifier_idx = np.argmax([max(scores) for scores in test_scores])
    best_classifier = classifiers[best_classifier_idx]

    # Plot decision boundaries
    for epoch in [0, np.argmax(test_scores[best_classifier_idx]), 99999]:
        plt.figure(figsize=(12, 5))
        for idx, (name, X, y) in enumerate([('Training', X_train, y_train), ('Test', X_test, y_test)]):
            plt.subplot(1, 2, idx + 1)
            plot_decision_boundary(lambda x: best_classifier.predict(x), X, y,
                                   title=f'Decision Boundary on {name} Set - Epoch {epoch}')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    for dataset, name in load_generated_datasets()[5:6]:
        experiment_4(dataset, name)