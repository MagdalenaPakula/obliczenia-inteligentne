from typing import List

import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.cluster import KMeans


def perform_clustering(dataset: ndarray, n_clusters: int) -> ndarray:
    # seed with a constant for consistent results
    RANDOM_STATE = 42

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(dataset)
    return kmeans.labels_


type __plottable = List[int | float] | ndarray | range


def plot_silhouette_scores(clusters: __plottable, silhouette_scores: __plottable, dataset_name: str) -> None:
    plt.plot(clusters, silhouette_scores, marker='o')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title(f'K-means experiment ({dataset_name})')
    plt.show()
