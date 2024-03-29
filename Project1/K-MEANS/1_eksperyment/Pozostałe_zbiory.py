import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy import ndarray
import utilities as util
from Project1.main import load_other_datasets

# TODO:
"""
STRONA - 6 - RAPORTU

Opis analizy pozostałych, rzeczywistych zbiorów danych, w której to zastosowane zostaną wnioski
z wcześniejszych eksperymentów. Wyniki tej analizy należy również uzasadnić poprzez odwołanie do
wartości miar uzyskiwanych na tych zbiorach (warto wykorzystać tabele i/lub wykresy i się do nich odwołać).
W przypadku zbioru Iris da się swoje wnioski podeprzeć wizualizacjami rzutów cech obiektów na dwuwymiarowe
przestrzenie wybranych kombinacji dwóch z nich.
"""


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


# Load Iris, Wine and Breast_cancer datasets
if __name__ == "__main__":
    datasets = load_other_datasets()

    for X, dataset_name in datasets:
        kmeans_experiment(X, dataset_name)
