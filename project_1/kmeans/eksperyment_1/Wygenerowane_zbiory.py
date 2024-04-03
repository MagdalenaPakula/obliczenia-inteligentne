from typing import List

import numpy as np
from numpy import ndarray
from sklearn.metrics import silhouette_score

from project_1.data import load_generated_datasets
from project_1.kmeans import perform_clustering
from project_1.visualization import plot_voronoi_diagram, plot_silhouette_scores

"""
 STRONA - 1 - RAPORTU
 
 Wyniki pierwszego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody K-Means.
 Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miary silhouette score przy zmieniającym
 się parametrze n-clusters oraz wizualizację klastrów (diagram Woronoja) dla najlepszego i najgorszego przypadku
 (wskazując, który to był przypadek i dlaczego).
"""


# Eksperyment z algorytmem kmeans
def kmeans_experiment(features: ndarray, dataset_name: str):
    cluster_range = range(2, 10)
    silhouette_scores: List[float] = []
    assigned_labels: List[ndarray] = []

    for n_clusters in cluster_range:
        labels = perform_clustering(features, n_clusters)
        silhouette_avg = silhouette_score(features, labels)
        silhouette_scores.append(silhouette_avg)
        assigned_labels.append(labels)

    plot_silhouette_scores(cluster_range, silhouette_scores, dataset_name)

    # Best and worst cases of silhouette scores
    best_cluster_index = np.argmax(silhouette_scores)
    worst_cluster_index = np.argmin(silhouette_scores)

    best_clustering = assigned_labels[best_cluster_index]
    worst_clustering = assigned_labels[worst_cluster_index]

    plot_voronoi_diagram(features[:, :2], best_clustering,
                         diagram_title=f'K-means clustering ({dataset_name}) - Best case')
    plot_voronoi_diagram(features[:, :2], worst_clustering,
                         diagram_title=f'K-means clustering ({dataset_name}) - Worst case')


if __name__ == "__main__":
    for dataset, name in load_generated_datasets():
        kmeans_experiment(dataset[:, :2], name)
