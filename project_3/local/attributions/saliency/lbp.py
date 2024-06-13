from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency

from project_2.part_1 import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.features.reduced_dimention.hog import hog_feature_extraction
from project_2.part_1.features.reduced_dimention.lbp import lbp_image
from project_3.models import get_model


def get_sample_data(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset_MNIST(transform=lbp_image)['test_dataset']
    return dataset[index]


def plot_lbp(lbp_image: torch.Tensor, axis: plt.Axes | None = None) -> plt.Figure | None:
    fig = None
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # Convert the 1D LBP tensor to a 2D grayscale image
    lbp_image = lbp_image.reshape(28, 28)

    axis.imshow(lbp_image.detach().numpy(), cmap='gray')
    axis.set_title('LBP')
    axis.axis('off')
    if fig is not None:
        return fig


def _main():
    model: MLP = get_model("MLP_mnist_lbp.pt")
    for i in range(10):
        img, label = get_sample_data(i + 10)
        img.requires_grad = True
        saliency = Saliency(model)
        model.eval()
        model.zero_grad()
        gradients = saliency.attribute(img.unsqueeze(0), target=label)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        fig.suptitle(f"Digit: {label}", fontsize=20)
        plot_lbp(img.detach(), axis=axes[0])
        axes[0].set_title('(LBP) representation of the image')
        plot_lbp(gradients.squeeze(0), axis=axes[1])
        axes[1].set_title('Saliency Map of the image')
        fig.show()


if __name__ == '__main__':
    _main()
