from typing import List

import numpy as np
from sklearn.metrics import silhouette_score

from project_1.DBSCAN import perform_clustering
from project_1.data import load_other_datasets
from project_1.visualization import plot_silhouette_scores_vs_eps, plot_voronoi_diagram

# TODO:
"""
STRONA - 6 - RAPORTU

Opis analizy pozostałych, rzeczywistych zbiorów danych, w której to zastosowane zostaną wnioski
z wcześniejszych eksperymentów. Wyniki tej analizy należy również uzasadnić poprzez odwołanie do
wartości miar uzyskiwanych na tych zbiorach (warto wykorzystać tabele i/lub wykresy i się do nich odwołać).
W przypadku zbioru Iris da się swoje wnioski podeprzeć wizualizacjami rzutów cech obiektów na dwuwymiarowe
przestrzenie wybranych kombinacji dwóch z nich.
"""


def dbscan_experiment(X, dataset_name):
    silhouette_scores = []
    assigned_labels = []

    eps_range = [0.1, 0.5, 1.0, 1.5, 2.0]

    for eps in eps_range:
        labels = perform_clustering(X, eps, min_samples=1)
        assigned_labels.append(labels)
        unique_labels = len(np.unique(labels))
        if unique_labels == 1 or unique_labels >= len(X):
            silhouette_scores.append(0)
        else:
            silhouette_scores.append(silhouette_score(X, labels))

    num_clusters = [len(np.unique(labels)) for labels in assigned_labels]

    plot_silhouette_scores_vs_eps(eps_range, silhouette_scores, num_clusters,
                                  plot_title=f'DBSCAN experiment silhouette scores ({dataset_name})')

    # Find best and worst EPS values
    best_case_index = np.argmax(silhouette_scores)
    worst_case_index = np.argmin(silhouette_scores)

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

    # Plot Voronoi diagram for best and worst cases
    plot_voronoi_diagram(X[:, features_to_plot], assigned_labels[best_case_index], None,
                         diagram_title=f'DBSCAN clustering ({dataset_name}) - Best case (EPS={eps_range[best_case_index]})')
    plot_voronoi_diagram(X[:, features_to_plot], assigned_labels[worst_case_index], None,
                         diagram_title=f'DBSCAN clustering ({dataset_name}) - Worst case (EPS={eps_range[worst_case_index]})')


if __name__ == "__main__":
    # Load other datasets
    datasets = load_other_datasets()

    # Performing experiments for each dataset
    for X, dataset_name in datasets:
        dbscan_experiment(X, dataset_name)
