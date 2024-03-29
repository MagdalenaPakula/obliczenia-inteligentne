import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from numpy import ndarray
import utilities as util

from Project1.main import load_generated_datasets

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
    cluster_range = range(2, 10)
    beta_values = [0.5, 1.0, 2.0]
    eps_range = [0.1, 0.5, 1.0, 1.5, 2.0]

    num_clusters = []
    rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = {beta: [] for beta in beta_values}

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(X)
        labels = dbscan.labels_
        unique_labels = np.unique(labels)
        num_clusters.append(len(unique_labels))

        rand_scores.append(adjusted_rand_score(y_true, labels))
        homogeneity_scores.append(homogeneity_score(y_true, labels))
        completeness_scores.append(completeness_score(y_true, labels))

        for beta in beta_values:
            v_measure_scores[beta].append(v_measure_score(y_true, labels, beta=beta))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('EPS Value')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(eps_range, rand_scores, marker='o', color=color, label='Adjusted rand Score')
    ax1.plot(eps_range, homogeneity_scores, marker='o',  color='green', label='Homogeneity score')
    ax1.plot(eps_range, completeness_scores, marker='o',  color='orange', label='Completeness score')
    for beta in beta_values:
        ax1.plot(eps_range, v_measure_scores[beta], marker='o', label=f'V-Measure score (beta={beta})')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    plt.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Number of Clusters', color=color)
    ax2.plot(eps_range, num_clusters, marker='o', color=color, label='Number of Clusters')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    ax1.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    plt.title(loc='center', label='DBSCAN Experiment ({dataset_name}) - clustering Evaluation Scores vs EPS')
    fig.tight_layout()
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

    for X, dataset_name in datasets[:1]:
        y_true = X[:, -1]
        dbscan_experiment(X[:, :-1], y_true, dataset_name)
