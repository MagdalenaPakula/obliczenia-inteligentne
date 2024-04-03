import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

from project_1.DBSCAN import perform_clustering
from project_1.data import load_generated_datasets
from project_1.visualization import plot_silhouette_scores_vs_eps, plot_voronoi_diagram

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
    assigned_labels = []

    eps_range = [0.1, 0.5, 1.0, 1.5, 2.0]

    for eps in eps_range:
        labels = perform_clustering(X, eps, min_samples=1)
        assigned_labels.append(labels)
        silhouette_scores.append(silhouette_score(X, labels))

    num_clusters = [len(np.unique(labels)) for labels in assigned_labels]

    plot_silhouette_scores_vs_eps(eps_range, silhouette_scores, num_clusters,
                                  plot_title=f'DBSCAN experiment silhouette scores ({dataset_name})')
    plt.savefig(f'{dataset_name[:-4]}_scores.png')
    plt.close()

    # Find best and worst EPS values
    best_case_index = np.argmax(silhouette_scores)
    worst_case_index = np.argmin(silhouette_scores)

    # Plot Voronoi diagram for best and worst cases
    plot_voronoi_diagram(X[:, :2], assigned_labels[best_case_index], None,
                         diagram_title=f'DBSCAN clustering ({dataset_name}) - Best case (EPS={eps_range[best_case_index]})')
    plt.savefig(f'{dataset_name[:-4]}_best.png')
    plt.close()
    plot_voronoi_diagram(X[:, :2], assigned_labels[worst_case_index], None,
                         diagram_title=f'DBSCAN clustering ({dataset_name}) - Worst case (EPS={eps_range[worst_case_index]})')
    plt.savefig(f'{dataset_name[:-4]}_worst.png')
    plt.close()


if __name__ == "__main__":
    # Load generated datasets
    datasets = load_generated_datasets()

    # Performing experiments for each dataset
    for X, dataset_name in datasets:
        dbscan_experiment(X, dataset_name)
