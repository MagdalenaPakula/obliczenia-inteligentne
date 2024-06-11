import numpy as np
import torch

from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
from project_2.part_2.models.kuba import MnistLargeModel
from project_3.local.attributions.saliency.mnist import get_sample_data
from project_3.models import get_model


def plot_original_and_attributions_with_colorbar(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    ax1.imshow(original_img, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Original image')

    # Attributions
    ax2.imshow(attributions_img, cmap='bwr', vmin=-attributions_img.max(), vmax=attributions_img.max())
    ax2.set_axis_off()
    ax2.set_title('Attributions')

    # Colorbar
    cbar = fig.colorbar(ax2.imshow(attributions_img, cmap='bwr'), ax=ax2)
    cbar.set_label('Attribution score')

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

        plot_original_and_attributions_with_colorbar(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
