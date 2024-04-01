import warnings
from typing import List

import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

from project_1.data import load_other_datasets
from project_1.kmeans import perform_clustering
from project_1.visualization import plot_other_scores, plot_voronoi_diagram

warnings.filterwarnings("ignore")

# TODO:
"""
STRONA - 6 - RAPORTU

Opis analizy pozostałych, rzeczywistych zbiorów danych, w której to zastosowane zostaną wnioski
z wcześniejszych eksperymentów. Wyniki tej analizy należy również uzasadnić poprzez odwołanie do
wartości miar uzyskiwanych na tych zbiorach (warto wykorzystać tabele i/lub wykresy i się do nich odwołać).
W przypadku zbioru Iris da się swoje wnioski podeprzeć wizualizacjami rzutów cech obiektów na dwuwymiarowe
przestrzenie wybranych kombinacji dwóch z nich.
"""


def kmeans_experiment(X, y_true, dataset_name):
    cluster_range = range(2, 10)
    beta_values = [0.5, 1.0, 2.0]

    rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = {beta: [] for beta in beta_values}

    assigned_labels = []

    for n_clusters in cluster_range:
        labels = perform_clustering(X, n_clusters)
        assigned_labels.append(labels)

        rand_scores.append(adjusted_rand_score(y_true, labels))
        homogeneity_scores.append(homogeneity_score(y_true, labels))
        completeness_scores.append(completeness_score(y_true, labels))

        for beta in beta_values:
            v_measure_scores[beta].append(v_measure_score(y_true, labels, beta=beta))

    metrics = {
        'Adjusted rand score': rand_scores,
        'Homogeneity score': homogeneity_scores,
        'Completeness score': completeness_scores
    }
    for beta in beta_values:
        metrics[f'V-measure score (beta={beta})'] = v_measure_scores[beta]

    plot_other_scores(cluster_range, metrics,
                      title=f'K-Means Experiment ({dataset_name}) - clustering evaluation scores')

    # Find best and worst cluster numbers for each metric
    metric_values = np.array(list(metrics.values()))
    mean_metrics = metric_values.mean(axis=0)
    best_cluster_index = mean_metrics.argmax()
    worst_cluster_index = mean_metrics.argmin()

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

    plot_voronoi_diagram(X[:, features_to_plot], assigned_labels[best_cluster_index], y_true,
                         diagram_title=f'K-means clustering ({dataset_name}) - Best case')
    plot_voronoi_diagram(X[:, features_to_plot], assigned_labels[worst_cluster_index], y_true,
                         diagram_title=f'K-means clustering ({dataset_name}) - Worst case')


if __name__ == "__main__":
    # Load generated datasets
    datasets = load_other_datasets()

    for X, dataset_name in datasets:
        y_true = X[:, -1]
        kmeans_experiment(X[:, :-1], y_true, dataset_name)
