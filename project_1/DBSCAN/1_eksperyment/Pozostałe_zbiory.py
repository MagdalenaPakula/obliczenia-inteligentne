import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from numpy import ndarray
import utilities as util

import numpy as np
from project_1.data import load_other_datasets

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
    num_clusters = []
    eps_range = [0.1, 0.5, 1.0, 1.5, 2.0]

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(X)
        labels = dbscan.labels_
        unique_labels = np.unique(labels)
        num_clusters.append(len(unique_labels))

        if len(unique_labels) > 1:
            silhouette_avg = silhouette_score(X, labels)
            silhouette_scores.append(silhouette_avg)
        else:
            print(f"For EPS={eps}, only one cluster is formed in {dataset_name}")
            silhouette_vals = silhouette_samples(X, labels)
            mean_silhouette = np.mean(silhouette_vals)
            silhouette_scores.append(mean_silhouette)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('EPS Value')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.plot(eps_range, silhouette_scores, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Number of Clusters', color=color)
    ax2.plot(eps_range, num_clusters, marker='o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # plt.grid(True)

    plt.title(f'DBSCAN experiment silhouette scores ({dataset_name})')
    fig.tight_layout()
    plt.show()

    # Find best and worst EPS values
    best_eps_index = silhouette_scores.index(max(silhouette_scores))
    worst_eps_index = silhouette_scores.index(min(silhouette_scores))

    # Plot Voronoi diagram for best and worst cases
    plot_voronoi_diagram(X[:, :2], eps_range[best_eps_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X[:, :2], eps_range[worst_eps_index], dataset_name, 'Worst Case')


def plot_voronoi_diagram(X: ndarray, eps: float, dataset_name: str, case: str):
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)

    y_pred = dbscan.labels_

    fig, ax = plt.subplots()
    util.plot_voronoi_diagram(X, None, y_pred, ax=ax)

    plt.title(f'DBSCAN clustering ({dataset_name}) - {case} (EPS={eps})')
    plt.show()


if __name__ == "__main__":
    # Load other datasets
    datasets = load_other_datasets()

    # Performing experiments for each dataset
    for X, dataset_name in datasets:
        dbscan_experiment(X, dataset_name)
