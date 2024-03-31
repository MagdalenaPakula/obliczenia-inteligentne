import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy import ndarray
import utilities as util

from project_1.main import load_generated_datasets

"""
 STRONA - 1 - RAPORTU
 
 Wyniki pierwszego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody K-Means.
 Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miary silhouette score przy zmieniającym
 się parametrze n-clusters oraz wizualizację klastrów (diagram Woronoja) dla najlepszego i najgorszego przypadku
 (wskazując, który to był przypadek i dlaczego).
"""


# Eksperyment z algorytmem K-MEANS
def kmeans_experiment(X, dataset_name):
    silhouette_scores = []
    cluster_range = range(2, 10)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)

    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title(f'K-means experiment ({dataset_name})')
    plt.show()

    # Best and worst cases of silhouette scores
    best_cluster_index = silhouette_scores.index(max(silhouette_scores))
    worst_cluster_index = silhouette_scores.index(min(silhouette_scores))

    # Visualizing for the best and worst using VORONOI
    plot_voronoi_diagram(X[:, :2], cluster_range[best_cluster_index], dataset_name, 'Best case')
    plot_voronoi_diagram(X[:, :2], cluster_range[worst_cluster_index], dataset_name, 'Worst case')


def plot_voronoi_diagram(X: ndarray, n_clusters: int, dataset_name: str, case: str):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    y_pred = kmeans.labels_

    fig, ax = plt.subplots()
    util.plot_voronoi_diagram(X, None, y_pred, ax=ax)  # Passing None for y_true

    plt.title(f'K-means clustering ({dataset_name}) - {case}')
    plt.show()


if __name__ == "__main__":
    datasets = load_generated_datasets()

    for X, dataset_name in datasets:
        kmeans_experiment(X, dataset_name)

    # for index, (X, dataset_name) in enumerate(datasets):
    #     if index == 5:  # Select the 6th file (index 5)
    #         kmeans_experiment(X, dataset_name)
    #         break  # Exit the loop after processing the 6th file
