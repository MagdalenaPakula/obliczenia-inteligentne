from typing import List, Optional, Dict, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial import Voronoi


def plot_voronoi_diagram(
        points: ndarray,
        predicted_labels: ndarray,
        actual_labels: Optional[ndarray] = None,
        diagram_title: Optional[str] = None,
        colormap: matplotlib.colors.Colormap = plt.get_cmap('brg'),
        point_size: int = 10) -> None:
    if points.shape[1] != 2:
        raise ValueError("Drawing Voronoi diagram requires 2D points")

    # add additional points to elliminate infinite regions
    augmented_points = np.array([*points, [999, 999], [-999, 999], [-999, -999], [999, -999]])
    vor = Voronoi(augmented_points)

    normalize = matplotlib.colors.Normalize(vmin=np.min(predicted_labels), vmax=np.max(predicted_labels))

    # draw voronoi regions
    for (i, point_region) in enumerate(vor.point_region):
        region = vor.regions[point_region]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            color = colormap(normalize(predicted_labels[i]))
            plt.fill(*zip(*polygon), alpha=0.4, color=color)

    # draw points
    if actual_labels is not None:
        plt.scatter(points[:, 0], points[:, 1], c=actual_labels, cmap=colormap, s=point_size)
    else:
        plt.scatter(points[:, 0], points[:, 1], color='black', s=point_size)

    plt.title(diagram_title)

    # zoom plot to area with points
    [min_x, min_y] = points.min(axis=0)
    [max_x, max_y] = points.max(axis=0)
    plt.xlim(min_x - 0.1, max_x + 0.1)
    plt.ylim(min_y - 0.1, max_y + 0.1)

    plt.show()


type __plottable = List[int | float] | ndarray | range


def plot_silhouette_scores(clusters: __plottable, silhouette_scores: __plottable, dataset_name: str) -> None:
    plt.plot(clusters, silhouette_scores, marker='o')
    plt.grid(axis='x', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title(f'K-means experiment ({dataset_name})')
    plt.show()


def plot_other_scores(clusters: __plottable, scores: Dict[str, __plottable], title: Optional[str] = None) -> None:
    plt.grid(axis='x', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score value')
    if title is not None:
        plt.title(title)
    for (name, score_values) in scores.items():
        plt.plot(clusters, score_values, marker='o', label=name)

    plt.legend()
    plt.show()


def plot_silhouette_scores_vs_eps(eps: __plottable, silhouette_scores: __plottable, n_clusters: __plottable,
                                  plot_title: Optional[str] = None) -> None:
    fig, silhouette_axis = plt.subplots()
    plt.grid(axis='x', linestyle='--')

    color = 'tab:red'
    silhouette_axis.set_xlabel('EPS Value')
    silhouette_axis.set_ylabel('Silhouette Score', color=color)
    silhouette_axis.plot(eps, silhouette_scores, marker='o', color=color)
    silhouette_axis.tick_params(axis='y', labelcolor=color)

    n_clusters_axis = silhouette_axis.twinx()

    color = 'tab:blue'
    n_clusters_axis.set_ylabel('Number of clusters', color=color)
    n_clusters_axis.plot(eps, n_clusters, marker='o', color=color)
    n_clusters_axis.tick_params(axis='y', labelcolor=color)

    if plot_title is not None:
        plt.title(plot_title)

    fig.show()


def plot_decision_boundary(classifier: Callable[[ndarray], ndarray],
                           features: ndarray,
                           labels: ndarray,
                           title: Optional[str] = None,
                           resolution: int = 100
                           ) -> None:
    if features.shape[1] != 2:
        raise ValueError("Plotting decision boundary requires 2D features")

    # Create a mesh of points for visualization
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Predict labels for each point in the mesh
    y_pred = classifier(np.c_[xx.ravel(), yy.ravel()])
    y_pred = y_pred.reshape(xx.shape)
    cmap_name = 'viridis'
    # Plot the decision boundary (using default colormap)
    plt.contourf(xx, yy, y_pred, alpha=0.5, cmap=cmap_name)
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=cmap_name)

    if title:
        plt.title(title)
