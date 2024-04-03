import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

from project_1.data import load_generated_datasets
from project_1.kmeans import perform_clustering
from project_1.visualization import plot_other_scores, plot_voronoi_diagram

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
    plt.savefig(f'{dataset_name[:-4]}_scores.png')
    plt.close()

    # Find best and worst cluster numbers for each metric
    metric_values = np.array(list(metrics.values()))
    mean_metrics = metric_values.mean(axis=0)
    best_cluster_index = mean_metrics.argmax()
    worst_cluster_index = mean_metrics.argmin()

    plot_voronoi_diagram(X[:, :2], assigned_labels[best_cluster_index], y_true,
                         diagram_title=f'K-means clustering ({dataset_name}) - Best case')
    plt.savefig(f'{dataset_name[:-4]}_best.png')
    plt.close()
    plot_voronoi_diagram(X[:, :2], assigned_labels[worst_cluster_index], y_true,
                         diagram_title=f'K-means clustering ({dataset_name}) - Worst case')
    plt.savefig(f'{dataset_name[:-4]}_worst.png')
    plt.close()


if __name__ == "__main__":
    datasets = load_generated_datasets()

    for X, dataset_name in datasets:
        y_true = X[:, -1]
        kmeans_experiment(X[:, :-1], y_true, dataset_name)
