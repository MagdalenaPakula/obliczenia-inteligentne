from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from project_1.data import load_generated_datasets


def knn_experiment(dataset, name):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)
    neighbor_range = range(1, 20)
    accuracy_train = []
    accuracy_test = []

    for n_neighbors in neighbor_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        accuracy_train.append(knn.score(X_train, y_train))
        accuracy_test.append(knn.score(X_test, y_test))

    # Wygenerowanie wykresu
    plt.plot(neighbor_range, accuracy_train, label="Zbiór treningowy")
    plt.plot(neighbor_range, accuracy_test, label="Zbiór testowy")
    plt.xlabel("Wartość parametru n_neighbours")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.title(f"Eksperyment dla zbioru danych: {name}")
    plt.show()


if __name__ == "__main__":
    datasets = load_generated_datasets()

    # Przeprowadzenie eksperymentu dla 2_2 i 2_3
    for dataset, name in datasets[4:6]:
        knn_experiment(dataset, name)
