from captum.attr import IntegratedGradients
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import visualization as viz

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models.kuba.cifar.large import CifarLargeModel, _CIFARLargeFeatureExtractor
from project_3.models import get_model


def get_sample_data_cifar(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dm = CIFAR10DataModule()
    dm.setup('')
    dataloader = dm.test_dataloader()

    images, labels = next(iter(dataloader))

    return images[index], labels[index]


def plot_original_and_attributions_with_blended_heatmap(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().cpu().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display the original image
    ax1.imshow(original_img, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Original image')

    # Visualize the blended heatmap
    viz.visualize_image_attr(attributions_img, original_image=original_img, method="blended_heat_map", sign="all",
                             show_colorbar=True, title="Overlayed Integrated Gradients", plt_fig_axis=(fig, ax2))

    fig.tight_layout()
    plt.show()


def _main():
    model: CifarLargeModel = get_model("kuba_cifar_large.pt")

    for i in range(10):
        img, label = get_sample_data_cifar(i)
        img.requires_grad = True

        ig = IntegratedGradients(model)
        model.eval()
        model.zero_grad()
        gradients = ig.attribute(img.unsqueeze(0), target=label.item())

        plot_original_and_attributions_with_blended_heatmap(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()