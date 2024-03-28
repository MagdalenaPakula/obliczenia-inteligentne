import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from numpy import ndarray
import utilities as util

from Project1.main import load_generated_datasets_with_labels, load_generated_datasets

"""
STRONA - 4 - RAPORTU
 
Wyniki drugiego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody DBSCAN.
Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości wartości miar adjusted rand score,
homogeneity score,  completeness score oraz V-measure score przy zmieniającym się parametrze eps oraz
wizualizację klastrów (diagram Woronoja z pokazanymi prawdziwymi etykietami obiektów)
dla najlepszego i najgorszego przypadku (wskazując, który to był przypadek i dlaczego).
W przypadku metody DBSCAN warto również wskazać na wykresie jaką liczbę klastrów  uzyskano
dla różnych wartości parametru eps.
"""

# DBSCAN Experiment Function
def dbscan_experiment(X, y_true, dataset_name):
    rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []
    num_clusters = []
    eps_range = [0.1, 0.5, 1.0, 1.5, 2.0]

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(X)
        labels = dbscan.labels_
        unique_labels = np.unique(labels)
        num_clusters.append(len(unique_labels))

        rand_scores.append(adjusted_rand_score(y_true, labels))
        homogeneity_scores.append(homogeneity_score(y_true, labels))
        completeness_scores.append(completeness_score(y_true, labels))
        v_measure_scores.append(v_measure_score(y_true, labels))

    # Plot evaluation scores for different eps values
    plt.plot(eps_range, rand_scores, marker='o', label='Adjusted rand Score')
    plt.plot(eps_range, homogeneity_scores, marker='o', label='Homogeneity score')
    plt.plot(eps_range, completeness_scores, marker='o', label='Completeness score')
    plt.plot(eps_range, v_measure_scores, marker='o', label='V-Measure score')
    plt.xlabel('EPS Value')
    plt.ylabel('Score')
    plt.title(f'DBSCAN Experiment ({dataset_name}) - Clustering Evaluation Scores vs EPS')
    plt.legend()
    plt.show()

    # Plot number of clusters obtained for different eps values
    plt.plot(eps_range, num_clusters, marker='o')
    plt.xlabel('EPS Value')
    plt.ylabel('Number of Clusters')
    plt.title(f'DBSCAN Experiment ({dataset_name}) - Number of Clusters vs EPS')
    plt.show()

    # Find best and worst EPS values
    best_eps_index = np.argmax(v_measure_scores)
    worst_eps_index = np.argmin(v_measure_scores)

    # Plot Voronoi diagram for best and worst cases
    plot_voronoi_diagram(X[:, :2], X[:, -1], eps_range[best_eps_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X[:, :2], X[:, -1], eps_range[worst_eps_index], dataset_name, 'Worst Case')


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
    for X, y_true, dataset_name in datasets:
        dbscan_experiment(X, y_true, dataset_name)
