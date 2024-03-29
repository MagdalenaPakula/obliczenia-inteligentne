import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from numpy import ndarray
import utilities as util

from Project1.main import load_generated_datasets

"""
STRONA - 3 - RAPORTU
 
Wyniki drugiego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody K-Means.
Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miar adjusted rand score,
homogeneity score,  completeness score oraz V-measure score przy zmieniającym się parametrze
n-clusters oraz wizualizację klastrów (diagram Woronoja z pokazanymi prawdziwymi etykietami obiektów)
dla najlepszego i najgorszego przypadku (wskazując, który to był przypadek i dlaczego).
"""


def kmeans_experiment(X, y_true, dataset_name):
    cluster_range = range(2, 10)
    beta_values = [0.5, 1.0, 2.0]

    rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = {beta: [] for beta in beta_values}

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_

        rand_scores.append(adjusted_rand_score(y_true, labels))
        homogeneity_scores.append(homogeneity_score(y_true, labels))
        completeness_scores.append(completeness_score(y_true, labels))

        for beta in beta_values:
            v_measure_scores[beta].append(v_measure_score(y_true, labels, beta=beta))

    plt.plot(cluster_range, rand_scores, marker='o', label='Adjusted Rand Score')
    plt.plot(cluster_range, homogeneity_scores, marker='o', label='Homogeneity score')
    plt.plot(cluster_range, completeness_scores, marker='o', label='Completeness score')
    for beta in beta_values:
        plt.plot(cluster_range, v_measure_scores[beta], marker='o', label=f'V-Measure score (beta={beta})')

    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title(f'K-Means Experiment ({dataset_name}) - clustering evaluation scores')
    plt.legend()
    plt.show()

    best_cluster_index = np.argmax(v_measure_scores)
    worst_cluster_index = np.argmin(v_measure_scores)

    plot_voronoi_diagram(X[:, :2], X[:, -1], cluster_range[best_cluster_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X[:, :2], X[:, -1], cluster_range[worst_cluster_index], dataset_name, 'Worst Case')


def plot_voronoi_diagram(X: ndarray, y_true: ndarray, n_clusters: int, dataset_name: str, case: str):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_

    y_pred = kmeans.labels_

    fig, ax = plt.subplots()
    util.plot_voronoi_diagram(X, y_true, y_pred, ax=ax)

    plt.title(f'K-means clustering ({dataset_name}) - {case}')
    plt.show()


if __name__ == "__main__":
    datasets = load_generated_datasets()

    for X, dataset_name in datasets[:1]:
        y_true = X[:, -1]
        kmeans_experiment(X[:, :-1], y_true, dataset_name)
