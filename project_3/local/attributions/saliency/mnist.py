from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency
from captum.attr import visualization as viz

from project_2.part_2.data import MNISTDataModule
from project_2.part_2.models.kuba import MnistLargeModel
from project_3.models import get_model


def get_sample_data(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dm = MNISTDataModule()
    dm.setup('')
    dataloader = dm.train_dataloader()

    images, labels = next(iter(dataloader))

    return images[index], labels[index]


def plot_original_and_attributions(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().numpy()

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
    model: MnistLargeModel = get_model("kuba_mnist_big.pt")

    for i in range(10):
        img, label = get_sample_data(i + 10)
        img.requires_grad = True

        saliency = Saliency(model)
        model.eval()
        model.zero_grad()
        gradients = saliency.attribute(img.unsqueeze(0), target=label.item())

        plot_original_and_attributions(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
