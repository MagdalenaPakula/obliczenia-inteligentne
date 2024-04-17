import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from project_1.classifiers.mlp_classifier import mlp_experiment
from project_1.data import load_generated_datasets
from project_1.visualization import plot_voronoi_diagram


def svm_experiment(X, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-1],
                                                        X[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    kernels = ['linear', 'rbf']  # Different kernel parameters
    best_svm = None
    best_svm_accuracy = 0

    for kernel in kernels:
        svm = SVC(kernel=kernel)
        svm.fit(X_train, y_train)

        accuracy_train = svm.score(X_train, y_train)
        if accuracy_train > best_svm_accuracy:
            best_svm_accuracy = accuracy_train
            best_svm = svm

        # Visualize decision boundary
        plot_voronoi_diagram(X_train[:, :2], svm.predict(X_train[:, :2]), y_train, f"SVM ({kernel}) - {dataset_name}")

        # Confusion matrix
        cm = confusion_matrix(y_train, svm.predict(X_train))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix for SVM ({kernel}) on {dataset_name}")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.show()

    return best_svm


if __name__ == "__main__":
    datasets = load_generated_datasets()

    for X, dataset_name in datasets[5:6]:
        best_svm = svm_experiment(X, dataset_name)
        best_mlp = mlp_experiment(X, dataset_name)
