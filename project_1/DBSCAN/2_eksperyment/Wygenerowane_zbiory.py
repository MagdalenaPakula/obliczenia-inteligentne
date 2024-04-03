import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

from project_1.DBSCAN import perform_clustering
from project_1.data import load_generated_datasets
from project_1.visualization import plot_voronoi_diagram

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
    beta_values = [0.5, 1.0, 2.0]
    eps_range = [0.1 * i + 0.05 for i in range(10)]
    assigned_labels = []

    num_clusters = []
    rand_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = {beta: [] for beta in beta_values}

    for eps in eps_range:
        labels = perform_clustering(X, eps, min_samples=1)
        assigned_labels.append(labels)
        unique_labels = np.unique(labels)
        num_clusters.append(len(unique_labels))

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

    plt.xlabel('EPS Value')
    plt.ylabel('Score')
    plt.xticks(eps_range, minor=True)
    plt.grid(True, axis='x', which='minor', linestyle='--')
    plt.grid(True, axis='y', which='major')
    for i, eps in enumerate(eps_range):
        plt.text(eps, np.array(list(metrics.values())).min(), str(num_clusters[i]))

    for (name, score_values) in metrics.items():
        plt.plot(eps_range, score_values, marker=None, label=name)

    plt.legend(loc='center right')

    plt.title(f'DBSCAN Experiment ({dataset_name}) - Clustering Evaluation Scores and Number of Clusters vs EPS',
              loc='center')
    plt.show()

    # Find best and worst eps values
    metric_values = np.array(list(metrics.values()))
    mean_metrics = metric_values.mean(axis=0)
    best_cluster_index = mean_metrics.argmax()
    worst_cluster_index = mean_metrics.argmin()

    plot_voronoi_diagram(X[:, :2], assigned_labels[best_cluster_index], y_true,
                         diagram_title=f'DBSCAN clustering ({dataset_name}) - Best case')
    plot_voronoi_diagram(X[:, :2], assigned_labels[worst_cluster_index], y_true,
                         diagram_title=f'DBSCAN clustering ({dataset_name}) - Worst case')


if __name__ == "__main__":
    # Load generated datasets
    datasets = load_generated_datasets()

    for X, dataset_name in datasets:
        y_true = X[:, -1]
        dbscan_experiment(X[:, :-1], y_true, dataset_name)
