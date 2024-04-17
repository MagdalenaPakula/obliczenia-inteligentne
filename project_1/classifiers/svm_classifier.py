import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from project_1.classifiers.mlp_classifier import mlp_experiment
from project_1.data import load_generated_datasets
from project_1.visualization import plot_voronoi_diagram, plot_decision_boundary


def svm_experiment(X, plot_title, train_size=0.2):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-1],
                                                        X[:, -1],
                                                        train_size=train_size,
                                                        test_size=0.2,
                                                        random_state=42)

    C_values = np.logspace(1, 4, 10)  # Exponential range from 31.62 to 1000

    accuracy_train = []
    accuracy_test = []
    classifiers = []

    for C in C_values:
        svm = SVC(kernel='rbf', C=C)
        svm.fit(X_train, y_train)

        accuracy_train.append(svm.score(X_train, y_train))
        accuracy_test.append(svm.score(X_test, y_test))
        classifiers.append(svm)

    # Plotting change in accuracy for different C values
    plt.plot(C_values, accuracy_train, label="Zbiór treningowy")
    plt.plot(C_values, accuracy_test, label="Zbiór testowy")
    plt.xlabel("Wartość parametru C")
    plt.ylabel("Dokładność")
    plt.xscale('log')  # Logarithmic scale for C values
    plt.legend()
    plt.title(f"Eksperyment dla zbioru danych: {plot_title}")
    plt.show()

    best_C_index = np.argmax(accuracy_test)
    best_C = C_values[best_C_index]

    svm_min = classifiers[0]
    svm_best = classifiers[best_C_index]
    svm_max = classifiers[-1]

    svm_min.fit(X_train, y_train)
    svm_best.fit(X_train, y_train)
    svm_max.fit(X_train, y_train)

    # Decision boundary and confusion matrix visualization
    for plot_title, svm, X, y in [("Minimalny klasyfikator SVM - Zbiór treningowy", svm_min, X_train, y_train),
                                  ("Minimalny klasyfikator SVM - Zbiór testowy", svm_min, X_test, y_test),
                                  ("Najlepszy klasyfikator SVM - Zbiór treningowy", svm_best, X_train, y_train),
                                  ("Najlepszy klasyfikator SVM - Zbiór testowy", svm_best, X_test, y_test),
                                  ("Maksymalny klasyfikator SVM - Zbiór treningowy", svm_max, X_train, y_train),
                                  ("Maksymalny klasyfikator SVM - Zbiór testowy", svm_max, X_test, y_test)]:

        plot_decision_boundary(lambda x: svm.predict(x), X, y, plot_title, resolution=200)


        cm = confusion_matrix(y, svm.predict(X))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Macierz pomyłek dla {plot_title}")
        plt.xlabel("Przewidziane etykiety")
        plt.ylabel("Rzeczywiste etykiety")
        plt.show()


if __name__ == "__main__":
    datasets = load_generated_datasets()

    for X, dataset_name in datasets[5:6]:
       svm_experiment(X, dataset_name)
