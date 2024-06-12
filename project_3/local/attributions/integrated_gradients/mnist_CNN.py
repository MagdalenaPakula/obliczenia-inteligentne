import numpy as np
import torch
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import slic

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

    viz.visualize_image_attr(attributions_img,
                             original_image=original_img,
                             method='blended_heat_map',
                             sign="all",
                             cmap="RdBu",
                             title="Overlayed Integrated Gradients",
                             plt_fig_axis=(fig, ax2))

    fig.tight_layout()
    plt.show()


def plot_original_and_attributions_with_slic(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().numpy()

    # Ensure original_img is RGB
    if original_img.shape[-1] == 1:
        original_img = np.concatenate([original_img] * 3, axis=-1)

    # Perform SLIC segmentation
    segments = slic(original_img, n_segments=100, compactness=10, sigma=1)
    slic_img = label2rgb(segments, original_img, kind='overlay')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # SLIC segmentation
    ax1.imshow(slic_img)
    ax1.set_axis_off()
    ax1.set_title('SLIC Segmentation')

    # Original image
    ax2.imshow(original_img)
    ax2.set_axis_off()
    ax2.set_title('Original image')

    # Attributions image with colorbar
    im = ax3.imshow(attributions_img, cmap="RdBu", vmin=-1, vmax=1)
    ax3.set_axis_off()
    ax3.set_title("Overlayed Integrated Gradients")

    # Adjust layout
    fig.tight_layout(pad=2.0)
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

        # plot_original_and_attributions_with_colorbar(img, gradients.squeeze(0))
        plot_original_and_attributions_with_slic(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
