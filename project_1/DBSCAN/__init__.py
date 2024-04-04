from numpy import ndarray
from sklearn.cluster import DBSCAN


def perform_clustering(dataset: ndarray, eps: float, min_samples: int) -> ndarray:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(dataset)
    return dbscan.labels_
