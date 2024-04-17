import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from project_1.data import load_generated_datasets
from project_1.visualization import plot_decision_boundary


def knn_experiment(X, plot_title, train_size=0.8):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-1],
                                                        X[:, -1],
                                                        train_size=train_size,
                                                        test_size=(1 - train_size),
                                                        random_state=42)
    # neighbor_range = range(130, 230)  # for 2_1 and 2_2
    neighbor_range = range(1, 10)  # for 2_3
    accuracy_train = []
    accuracy_test = []
    classifiers = []

    for n_neighbors in neighbor_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        accuracy_train.append(knn.score(X_train, y_train))
        accuracy_test.append(knn.score(X_test, y_test))
        classifiers.append(knn)

    # Wygenerowanie wykresu obrazujący zmianę wartości accuracy na zbiorach treningowym i
    # testowym przy zmieniającym się parametrze n_neighbours (1ST PART)
    plt.plot(neighbor_range, accuracy_train, label="Zbiór treningowy")
    plt.plot(neighbor_range, accuracy_test, label="Zbiór testowy")
    plt.xlabel("Wartość parametru n_neighbours")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.title(f"Eksperyment dla zbioru danych: {plot_title}")
    plt.show()

    # Utworzenie klasyfikatorów KNN dla wybranych wartości parametru
    knn_min = classifiers[0]
    knn_best = classifiers[np.argmax(accuracy_test)]
    knn_max = classifiers[-1]

    # Trening klasyfikatorów
    knn_min.fit(X_train, y_train)
    knn_best.fit(X_train, y_train)
    knn_max.fit(X_train, y_train)

    # Wizualizacja granicy decyzyjnej
    for plot_title, knn, X, y in [("Minimalny klasyfikator KNN - Zbiór treningowy", knn_min, X_train, y_train),
                                  ("Minimalny klasyfikator KNN - Zbiór testowy", knn_min, X_test, y_test),
                                  ("Najlepszy klasyfikator KNN - Zbiór treningowy", knn_best, X_train, y_train),
                                  ("Najlepszy klasyfikator KNN - Zbiór testowy", knn_best, X_test, y_test),
                                  ("Maksymalny klasyfikator KNN - Zbiór treningowy", knn_max, X_train, y_train),
                                  ("Maksymalny klasyfikator KNN - Zbiór testowy", knn_max, X_test, y_test)]:
        # Wizualizacja granicy decyzyjnej
        # plot_voronoi_diagram(X[:, :2], knn.predict(X[:, :2]), X[:, -1], dataset_name)
        # plot_voronoi_diagram(X, knn.predict(X), y, plot_title)
        plot_decision_boundary(lambda x: knn.predict(x), X, y, plot_title, resolution=200)

        # Dodatkowo stworz kod aby przy każdej wizualizacji należy pokazać jak wygląda macierz pomyłek.
        # Obliczenie i wyświetlenie macierzy pomyłek
        cm = confusion_matrix(y, knn.predict(X))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Macierz pomyłek dla {plot_title}")
        plt.xlabel("Przewidziane etykiety")
        plt.ylabel("Rzeczywiste etykiety")
        plt.show()


def find_best_n_neighbors(accuracy_test):
    # Znajdowanie indeksu najlepszej wartości accuracy na zbiorze testowym
    best_idx = np.argmax(accuracy_test)
    return best_idx + 1


if __name__ == "__main__":
    datasets = load_generated_datasets()

    # Przeprowadzenie eksperymentu 2_3 (2_1 i 2_2 trzeba poprawic)
    for X, dataset_name in datasets[4:]:
        knn_experiment(X, dataset_name)
