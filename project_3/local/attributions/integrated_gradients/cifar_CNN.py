import numpy as np
from captum.attr import IntegratedGradients
from typing import Tuple
import torch
from matplotlib import pyplot as plt

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models.kuba.cifar.large import CifarLargeModel, _CIFARLargeFeatureExtractor
# from project_3.local.attributions.saliency.mnist import plot_original_and_attributions
from project_3.models import get_model


def get_sample_data_cifar(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dm = CIFAR10DataModule()
    dm.setup('')
    dataloader = dm.test_dataloader()

    images, labels = next(iter(dataloader))

    return images[index], labels[index]


def plot_original_and_attributions(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = (original.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    attributions_img: np.ndarray = (attributions.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)

    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].imshow(original_img)
    axis[0].set_axis_off()
    axis[0].set_title('Original image')
    axis[1].imshow(attributions_img)
    axis[1].set_axis_off()
    axis[1].set_title('Attributions')
    fig.show()


def _main():
    model: CifarLargeModel = get_model("kuba_cifar_large.pt")

    for i in range(10):
        img, label = get_sample_data_cifar(i)
        img.requires_grad = True

        ig = IntegratedGradients(model)
        model.eval()
        model.zero_grad()
        gradients = ig.attribute(img.unsqueeze(0), target=label.item())

        plot_original_and_attributions(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
