from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt

from project_2.part_2.models.kuba import MnistLargeModel
from project_3.local.attributions.saliency.mnist import get_sample_data
from project_3.models import get_model

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_original_and_attributions(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().numpy()

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the original image
    ax1.imshow(original_img, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Original image')

    # Plot the attributions
    ax2.imshow(attributions_img, cmap='bwr', vmin=-attributions_img.max(), vmax=attributions_img.max())
    ax2.set_axis_off()
    ax2.set_title('Attributions')

    # Add a colorbar
    cbar = fig.colorbar(ax2.imshow(attributions_img, cmap='bwr'), ax=ax2)
    cbar.set_label('Attribution score')

    # Plot a bar chart of the attribution values
    attribution_values = attributions_img.reshape(-1)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(range(len(attribution_values)), attribution_values)
    ax3.set_xlabel('Pixel index')
    ax3.set_ylabel('Attribution score')
    ax3.set_title('Attribution values')

    fig.tight_layout()
    plt.show()
def _main():
    model: MnistLargeModel = get_model("kuba_mnist_big.pt")

    for i in range(10):
        img, label = get_sample_data(i)
        img.requires_grad = True

        ig = IntegratedGradients(model)
        model.eval()
        model.zero_grad()
        gradients = ig.attribute(img.unsqueeze(0), target=label.item())

        plot_original_and_attributions(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
