import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from scipy.spatial import Voronoi, voronoi_plot_2d

from Project1.main import load_generated_datasets_with_labels

"""
STRONA - 3 - RAPORTU
 
Wyniki drugiego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody K-Means.
Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miar adjusted rand score,
homogeneity score,  completeness score oraz V-measure score przy zmieniającym się parametrze
n-clusters oraz wizualizację klastrów (diagram Woronoja z pokazanymi prawdziwymi etykietami obiektów)
dla najlepszego i najgorszego przypadku (wskazując, który to był przypadek i dlaczego).
"""


def kmeans_experiment(X, y_true, dataset_name):
    rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []
    cluster_range = range(2, 10)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_

        rand_scores.append(adjusted_rand_score(y_true, labels))
        homogeneity_scores.append(homogeneity_score(y_true, labels))
        completeness_scores.append(completeness_score(y_true, labels))
        v_measure_scores.append(v_measure_score(y_true, labels))

    plt.plot(cluster_range, rand_scores, marker='o', label='Adjusted rand Score')
    plt.plot(cluster_range, homogeneity_scores, marker='o', label='Homogeneity score')
    plt.plot(cluster_range, completeness_scores, marker='o', label='Completeness score')
    plt.plot(cluster_range, v_measure_scores, marker='o', label='V-Measure score')

    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title(f'K-Means Experiment ({dataset_name}) - Clustering evaluation scores')
    plt.legend()
    plt.show()

    best_cluster_index = np.argmax(v_measure_scores)
    worst_cluster_index = np.argmin(v_measure_scores)

    # Load true labels
    plot_voronoi_diagram(X, y_true, cluster_range[best_cluster_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X, y_true, cluster_range[worst_cluster_index], dataset_name, 'Worst Case')


def plot_voronoi_diagram(X, y_true, n_clusters, dataset_name, case):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_

    if len(cluster_centers) >= 3:  # Ensure there are enough points to construct the initial simplex
        vor = Voronoi(cluster_centers)

        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=2)

        for label in np.unique(y_true):
            ax.plot(X[y_true == label, 0], X[y_true == label, 1], 'o', label=f'True Label {int(label)}')

        ax.legend()
        ax.grid(True)
        plt.title(f'Voronoi Diagram ({dataset_name}) - {case}')
        plt.show()
    else:
        print("Not enough points to construct the diagram.")


# Load generated datasets
datasets = load_generated_datasets_with_labels()

# Performing experiments for each dataset
for X, y_true, dataset_name in datasets:
    kmeans_experiment(X, y_true, dataset_name)
