from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency
from captum.attr import visualization as viz

from project_2.part_1 import MLP
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.features.reduced_dimention.lbp import lbp_feature_extraction
from project_3.models import get_model


def get_sample_data(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset_MNIST(transform=lbp_feature_extraction)['test_dataset']
    return dataset[index]


def plot_original_and_attributions(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 2, "Original tensor does not have 2 dimensions (height, width)"
    assert attributions.dim() == 2, "Attributions tensor does not have 2 dimensions (height, width)"

    original_img: np.ndarray = original.detach().unsqueeze(2).numpy()
    attributions_img: np.ndarray = attributions.detach().unsqueeze(2).numpy()

    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(original_img, cmap='gray')
    axis[0].set_axis_off()
    axis[0].set_title('Original image')
    viz.visualize_image_attr(attributions_img, original_img,
                             method='blended_heat_map',
                             sign='absolute_value',
                             plt_fig_axis=(fig, axis[1]),
                             use_pyplot=False)
    axis[1].set_title('Overlayed attributions')
    fig.show()


def _main():
    model: MLP = get_model("MLP_mnist_lbp_12.pt")

    for i in range(10):
        img, label = get_sample_data(i + 10)
        img.requires_grad = True

        saliency = Saliency(model)
        model.eval()
        model.zero_grad()
        gradients = saliency.attribute(img.unsqueeze(0), target=label)

        plot_original_and_attributions(img.detach().reshape(28, 28), gradients.squeeze(0).reshape(28, 28))


if __name__ == '__main__':
    _main()
