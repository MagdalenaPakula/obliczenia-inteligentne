from numpy import ndarray
from sklearn.cluster import KMeans


def perform_clustering(dataset: ndarray, n_clusters: int) -> ndarray:
    # seed with a constant for consistent results
    RANDOM_STATE = 42

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(dataset)
    return kmeans.labels_
