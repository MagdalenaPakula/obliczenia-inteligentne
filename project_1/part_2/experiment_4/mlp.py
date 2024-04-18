import pathlib
import warnings
from copy import deepcopy
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from project_1.data import load_generated_datasets
from project_1.visualization import plot_decision_boundary

warnings.filterwarnings("ignore")

NUM_EPOCHS = 100_000


def calculate_score(train_score, test_score):
    return test_score + 1e-5 * train_score


def visualize_training_over_time(dataset, train_size):
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=train_size,
                                                        test_size=0.2,
                                                        random_state=42)

    all_classes = np.unique(dataset[:, -1])
    epochs = range(1, NUM_EPOCHS + 1)
    train_scores = []
    test_scores = []

    classifier = MLPClassifier(hidden_layer_sizes=(5,),  # changed to the most optimal num_neurons from exp 2 and 3
                               activation='relu',
                               max_iter=NUM_EPOCHS,
                               tol=0,
                               n_iter_no_change=100000,
                               solver='sgd',
                               random_state=42)

    classifiers: Dict[str, MLPClassifier] = dict()

    for epoch in epochs:
        if epoch % 1_000 == 0:
            print(f'Epoch {epoch}')

        classifier.partial_fit(X_train, y_train, classes=all_classes)

        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)

        if epoch == epochs[0]:
            classifiers['first'] = deepcopy(classifier)
        else:
            # can only calculate best score if there are entries
            best_score = np.max([calculate_score(tr, tst) for (tr, tst) in zip(train_scores, test_scores)])
            if calculate_score(train_score, test_score) > best_score:
                classifiers['best'] = deepcopy(classifier)
        if epoch == epochs[-1]:
            classifiers['last'] = deepcopy(classifier)

        train_scores.append(train_score)
        test_scores.append(test_score)

    # Plot accuracy over time
    plt.plot(epochs, train_scores, label='train set')
    plt.plot(epochs, test_scores, label='test set')
    plt.grid(linestyle='--', axis='x')
    plt.legend(loc='best')
    plt.title(f'MLP accuracy over time, for training set size of {train_size}')
    pathlib.Path(f'exp_4/set_{train_size}').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'exp_4/set_{train_size}/accuracies.png')
    plt.show()
    plt.close()

    for case, classifier in classifiers.items():
        pathlib.Path(f'exp_4/set_{train_size}/{case}').mkdir(parents=True, exist_ok=True)

        plot_decision_boundary(lambda x: classifier.predict(x),
                               X_train, y_train, title=f'Decision boundary for {case} epoch on train set')
        plt.savefig(f'exp_4/set_{train_size}/{case}/train_boundary.png')
        plt.show()

        plot_decision_boundary(lambda x: classifier.predict(x),
                               X_test, y_test, title=f'Decision boundary for {case} epoch on test set')
        plt.savefig(f'exp_4/set_{train_size}/{case}/test_boundary.png')
        plt.show()


def test_different_starting_configurations(dataset, train_size):
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=train_size,
                                                        test_size=0.2,
                                                        random_state=42)

    all_classes = np.unique(dataset[:, -1])
    epochs = range(1, NUM_EPOCHS + 1)

    random_seeds = [12, 31, 16, 2353, 314, 6523, 634, 23, 802, 134]

    print(f'- Training set size: {train_size}')
    print('Seed, '
          'train_accuracy_first, train_accuracy_best, train_best_epoch, train_accuracy_end, '
          'test_accuracy_first, test_accuracy_best, test_best_epoch, test_accuracy_end')
    for seed in random_seeds:
        classifier = MLPClassifier(hidden_layer_sizes=(5,),  # changed to the most optimal num_neurons from exp 2 and 3
                                   activation='relu',
                                   max_iter=NUM_EPOCHS,
                                   tol=0,
                                   n_iter_no_change=100000,
                                   solver='sgd',
                                   random_state=seed)

        train_scores = []
        test_scores = []

        for _ in epochs:
            classifier.partial_fit(X_train, y_train, classes=all_classes)

            train_score = classifier.score(X_train, y_train)
            test_score = classifier.score(X_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

        train_accuracy_first = train_scores[0]
        train_accuracy_best = np.max(train_scores)
        train_best_epoch = np.argmax(train_scores) + 1
        train_accuracy_end = train_scores[-1]

        test_accuracy_first = test_scores[0]
        test_accuracy_best = np.max(test_scores)
        test_best_epoch = np.argmax(test_scores) + 1
        test_accuracy_end = test_scores[-1]

        print(f'{seed}, '
              f'{train_accuracy_first}, {train_accuracy_best}, {train_best_epoch}, {train_accuracy_end}, '
              f'{test_accuracy_first}, {test_accuracy_best}, {test_best_epoch}, {test_accuracy_end}')


if __name__ == '__main__':
    for dataset, name in load_generated_datasets()[5:6]:
        # For both train sizes
        for train_size in [0.2, 0.8]:
            print(f'Starting experiment with train set size of {train_size}')
            visualize_training_over_time(dataset, train_size)
            test_different_starting_configurations(dataset, train_size)
