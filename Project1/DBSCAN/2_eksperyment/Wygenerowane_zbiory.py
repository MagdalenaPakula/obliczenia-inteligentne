import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

from Project1.main import load_generated_datasets_with_labels

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
    plot_voronoi_diagram(X, None, eps_range[best_eps_index], dataset_name, 'Best Case')
    plot_voronoi_diagram(X, None, eps_range[worst_eps_index], dataset_name, 'Worst Case')


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
datasets = load_generated_datasets_with_labels()

# Performing experiments for each dataset
for X, y_true, dataset_name in datasets:
    dbscan_experiment(X, y_true, dataset_name)
