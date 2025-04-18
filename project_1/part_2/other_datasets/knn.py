import warnings
from typing import Dict, Tuple, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from project_1.data import load_generated_datasets, load_other_datasets
from project_1.visualization import plot_decision_boundary

type Classifier = KNeighborsClassifier

warnings.filterwarnings("ignore")


def experiment_2(dataset, dataset_name):
    features_to_plot: List[int]
    match dataset_name:
        case 'Iris Dataset':
            features_to_plot = [0, 1]  # Selecting features 0 and 1 for Iris Dataset
        case 'Wine Dataset':
            features_to_plot = [0, 2]  # Selecting features 0 and 2 for Wine Dataset
        case 'Breast Cancer Wisconsin Dataset':
            features_to_plot = [3, 4]  # Selecting features 3 and 4 for Breast Cancer Wisconsin Dataset
        case _:
            raise ValueError(f'Unknown dataset name {dataset_name}')

        # Split dataset and select specific features for visualization
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, features_to_plot],
                                                        dataset[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    neighbors_values = range(1, 10)
    classifiers: List[Classifier] = []
    train_scores = []
    test_scores = []
    for n_neighbors in neighbors_values:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(X_train, y_train)

        train_scores.append(classifier.score(X_train, y_train))
        test_scores.append(classifier.score(X_test, y_test))
        classifiers.append(classifier)

    plt.plot(neighbors_values, train_scores, label='train set')
    plt.plot(neighbors_values, test_scores, label='test set')
    plt.grid(linestyle='--', axis='x')
    plt.xlabel('Hidden layer size')
    plt.ylabel('score')
    plt.legend(loc='best')
    plt.title(f'KNN on dataset {dataset_name}')
    plt.show()
    plt.close()

    data: Dict[str, Tuple[ndarray, ndarray, List[float]]] = {'train': (X_train, y_train, train_scores),
                                                             'test': (X_test, y_test, test_scores)}

    min_classifier = classifiers[0]
    max_classifier = classifiers[-1]
    for set_choice, (features, labels, scores) in data.items():
        best_classifier = classifiers[np.argmax(scores)]

        selected_classifiers: Dict[str, Classifier] = {'min': min_classifier,
                                                       'best': best_classifier,
                                                       'max': max_classifier}
        for classifier_choice, classifier in selected_classifiers.items():
            plot_decision_boundary(lambda x: classifier.predict(x), features, labels,
                                   title=f'{classifier_choice} KNN on {dataset_name} - {set_choice} set')
            plt.show()
            plt.close()

            cm = confusion_matrix(y_train, classifier.predict(X_train))
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title(f"Confusion Matrix for {classifier_choice} KNN on {dataset_name} - {set_choice} set")
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            plt.show()


if __name__ == '__main__':
    for dataset, name in load_other_datasets():
        experiment_2(dataset, name)
