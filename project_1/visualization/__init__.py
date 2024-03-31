from typing import Optional

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
