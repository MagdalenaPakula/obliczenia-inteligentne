import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from project_1.data import load_generated_datasets
from project_1.visualization import plot_decision_boundary


def experiment_1(dataset: ndarray, dataset_name: str):
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    kernels = ['linear', 'rbf']

    for kernel in kernels:
        train_scores = []
        test_scores = []
        C_values = [i * 0.02 for i in range(1, 51)]
        classifiers = []
        for C in C_values:
            svm = SVC(kernel=kernel, C=C)
            svm.fit(X_train, y_train)
            classifiers.append(svm)

            train_score = svm.score(X_train, y_train)
            train_scores.append(train_score)
            test_score = svm.score(X_test, y_test)
            test_scores.append(test_score)

        plt.plot(C_values, train_scores, label='train set')
        plt.plot(C_values, test_scores, label='test set')
        plt.grid(linestyle='--', axis='x')
        plt.xlabel('C')
        plt.ylabel('score')
        plt.legend(loc='best')
        plt.title(f'SVM scores with kernel {kernel} on dataset {dataset_name}')
        plt.show()

        best_classifier = classifiers[np.argmax(
            [test_score + 1e-5 * train_score for (test_score, train_score) in zip(test_scores, train_scores)])]

        plot_decision_boundary(lambda x: best_classifier.predict(x), dataset[:, :2], dataset[:, -1],
                               title=f'SVM with kernel {kernel}, dataset {dataset_name}')
        plt.show()


if __name__ == '__main__':
    for dataset, name in load_generated_datasets()[3:]:
        experiment_1(dataset, name)
