import itertools
import threading
import time
from typing import List

import numpy as np
from numpy import ndarray
from sklearn.metrics import silhouette_score

from project_1.data import load_other_datasets
from project_1.kmeans.eksperyment_1 import perform_clustering, plot_silhouette_scores
from project_1.visualization import plot_voronoi_diagram

# TODO:
"""
STRONA - 6 - RAPORTU

Opis analizy pozostałych, rzeczywistych zbiorów danych, w której to zastosowane zostaną wnioski
z wcześniejszych eksperymentów. Wyniki tej analizy należy również uzasadnić poprzez odwołanie do
wartości miar uzyskiwanych na tych zbiorach (warto wykorzystać tabele i/lub wykresy i się do nich odwołać).
W przypadku zbioru Iris da się swoje wnioski podeprzeć wizualizacjami rzutów cech obiektów na dwuwymiarowe
przestrzenie wybranych kombinacji dwóch z nich.
"""


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

    features_to_plot: List[int]
    match dataset_name:
        case 'Iris Dataset':
            features_to_plot = [0, 2]
        case 'Wine Dataset':
            features_to_plot = [0, 1]
        case 'Breast Cancer Wisconsin Dataset':
            features_to_plot = [3, 24]
        case _:
            raise ValueError(f'Unknown dataset name {dataset_name}')

    plot_voronoi_diagram(features[:, features_to_plot], best_clustering,
                         diagram_title=f'K-means clustering ({dataset_name}) - Best case')
    plot_voronoi_diagram(features[:, features_to_plot], worst_clustering,
                         diagram_title=f'K-means clustering ({dataset_name}) - Worst case')


if __name__ == "__main__":
    for dataset, name in load_other_datasets():
        kmeans_experiment(dataset[:, :-1], name)
