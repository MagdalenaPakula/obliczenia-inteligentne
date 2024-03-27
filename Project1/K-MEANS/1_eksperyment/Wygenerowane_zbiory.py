from typing import List

import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import utilities as util
from Project1.main import load_generated_datasets

"""
 STRONA - 1 - RAPORTU
 
 Wyniki pierwszego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody K-Means.
 Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miary silhouette score przy zmieniającym
 się parametrze n-clusters oraz wizualizację klastrów (diagram Woronoja) dla najlepszego i najgorszego przypadku
 (wskazując, który to był przypadek i dlaczego).
"""


# Eksperyment z algorytmem K-MEANS
def kmeans_experiment(X: ndarray, dataset_name: str) -> None:
    silhouette_scores: List[float] = []
    cluster_range: range = range(2, 10)

    for n_clusters in cluster_range:
        kmeans: KMeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        labels: ndarray = kmeans.labels_
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)

    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title(f'K-means experiment silhouette scores ({dataset_name})')
    plt.show()

    # Best and worst cases of silhouette scores
    best_cluster_index = silhouette_scores.index(max(silhouette_scores))
    worst_cluster_index = silhouette_scores.index(min(silhouette_scores))

    # Visualizing for the best and worst using VORONOI
    plot_voronoi_diagram(X[:, :2], X[:, -1], cluster_range[best_cluster_index], dataset_name, 'Best case')
    plot_voronoi_diagram(X[:, :2], X[:, -1], cluster_range[worst_cluster_index], dataset_name, 'Worst case')


def plot_voronoi_diagram(X: ndarray, y_true: ndarray, n_clusters: int, dataset_name: str, case: str):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    y_pred = kmeans.labels_

    fig, ax = plt.subplots()
    util.plot_voronoi_diagram(X, y_true, y_pred, ax=ax)

    plt.title(f'K-means clustering ({dataset_name}) - {case}')
    plt.show()


if __name__ == "__main__":
    datasets = load_generated_datasets()

    for X, dataset_name in datasets:
        kmeans_experiment(X, dataset_name)
