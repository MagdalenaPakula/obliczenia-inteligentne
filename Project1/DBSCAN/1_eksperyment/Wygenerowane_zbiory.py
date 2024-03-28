import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from numpy import ndarray
import utilities as util

import numpy as np

from Project1.main import load_generated_datasets

"""
 STRONA - 2 - RAPORTU
 
 Wyniki pierwszego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody DBSCAN.
 Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miary silhouette score rzy zmieniającym
 się parametrze eps oraz wizualizację klastrów (diagram Woronoja) dla najlepszego i najgorszego przypadku
 (wskazując, który to był przypadek i dlaczego). W przypadku metody DBSCAN warto również wskazać na wykresie
 jaką liczbę klastrów  uzyskano dla różnych wartości parametru eps. Przykładowy wykres:
"""


# Eksperyment z algorytmem DBSCAN
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
            # Compute silhouette scores for individual samples
            silhouette_vals = silhouette_samples(X, labels)
            mean_silhouette = np.mean(silhouette_vals)
            silhouette_scores.append(mean_silhouette)

    # Plot silhouette scores for different eps values
    plt.plot(eps_range, silhouette_scores, marker='o')
    plt.xlabel('EPS Value')
    plt.ylabel('Silhouette score')
    plt.title(f'DBSCAN Experiment ({dataset_name}) - Silhouette Score vs EPS')
    plt.show()

    # Plot number of clusters obtained for different eps values
    plt.plot(eps_range, num_clusters, marker='o')
    plt.xlabel('EPS Value')
    plt.ylabel('Number of Clusters')
    plt.title(f'DBSCAN Experiment ({dataset_name}) - Number of Clusters vs EPS')
    plt.show()

    # Find best and worst EPS values
    best_eps_index = silhouette_scores.index(max(silhouette_scores))
    worst_eps_index = silhouette_scores.index(min(silhouette_scores))

    # Plot Voronoi diagram for best and worst cases
    plot_voronoi_diagram(X[:, :2], X[:, -1],  eps_range[best_eps_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X[:, :2], X[:, -1],  eps_range[worst_eps_index], dataset_name, 'Worst Case')


def plot_voronoi_diagram(X: ndarray, y_true: ndarray, eps: float, dataset_name: str, case: str):
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)

    y_pred = dbscan.labels_

    fig, ax = plt.subplots()
    util.plot_voronoi_diagram(X, y_true, y_pred, ax=ax)

    plt.title(f'DBSCAN clustering ({dataset_name}) - {case} (EPS={eps})')
    plt.show()

if __name__ == "__main__":
    # Load generated datasets
    datasets = load_generated_datasets()

    # Performing experiments for each dataset
    for X, dataset_name in datasets:
        dbscan_experiment(X, dataset_name)
