from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency

from project_2.part_1 import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.features.reduced_dimention.hog import hog_feature_extraction
from project_3.models import get_model


def get_sample_data(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset_MNIST(transform=hog_feature_extraction)['test_dataset']
    return dataset[index]


def plot_hog(attributions: torch.Tensor, axis: plt.Axes | None = None) -> plt.Figure | None:
    fig = None
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for i in range(4):
        ax: plt.Axes = axis
        points = attributions[i * 9: i * 9 + 9]
        X = np.zeros(9) + (i % 2)
        Y = np.zeros(9) + (i // 2)
        angles = np.linspace(0, 2 * np.pi, 10)[:-1] + (np.pi / 2)
        U = np.cos(angles)
        V = np.sin(angles)
        ax.quiver(X, Y, U, V, points, cmap='Blues', pivot='mid', scale=3,
                  headaxislength=0,
                  headlength=0,
                  headwidth=0)

    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(-0.5, 1.5)
    if fig is not None:
        return fig


def _main():
    model: MLP = get_model("MLP_mnist_hog.pt")

    for i in range(10):
        img, label = get_sample_data(i + 10)
        img.requires_grad = True

        saliency = Saliency(model)
        model.eval()
        model.zero_grad()
        gradients = saliency.attribute(img.unsqueeze(0), target=label)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        fig.suptitle(f"Digit: {label}", fontsize=20)

        plot_hog(img.detach(), axis=axes[0])
        axes[0].set_title('HOG')

        plot_hog(gradients.squeeze(0), axis=axes[1])
        axes[1].set_title('Attributions')
        fig.show()


if __name__ == '__main__':
    _main()
