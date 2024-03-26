import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

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


def perform_clustering_experiments(dataset, dataset_name):
    dbscan_experiment(dataset, dataset_name)


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
    plot_voronoi_diagram(X[:, :2], None, eps_range[best_eps_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X[:, :2], None, eps_range[worst_eps_index], dataset_name, 'Worst Case')


def plot_voronoi_diagram(X, y_true, eps, dataset_name, case):
    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(X)
    vor = Voronoi(X)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=2)

    if len(np.unique(dbscan.labels_)) > 1:
        for label in np.unique(dbscan.labels_):
            ax.plot(X[dbscan.labels_ == label, 0], X[dbscan.labels_ == label, 1],
                    'o', label=f'Cluster {int(label)}')

        # Plot true labels if provided
        if y_true is not None:
            for label in np.unique(y_true):
                ax.plot(X[y_true == label, 0], X[y_true == label, 1], 'o', markersize=5,
                        label=f'True Label {int(label)}', markeredgecolor='black')

        ax.legend()
        ax.grid(True)
        plt.title(f'Voronoi Diagram ({dataset_name}) - {case} (EPS={eps})')
        plt.show()
    else:
        print(f"No clusters found for EPS={eps} in {dataset_name}")


# Load generated datasets
datasets = load_generated_datasets()

# Performing experiments for each dataset
for X, dataset_name in datasets:
    perform_clustering_experiments(X, dataset_name)
