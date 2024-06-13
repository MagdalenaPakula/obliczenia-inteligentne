from typing import Tuple
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from skimage.color import label2rgb
from skimage.segmentation import slic
from sklearn.preprocessing import MinMaxScaler

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models.kuba.cifar.large import CifarLargeModel, _CIFARLargeFeatureExtractor
from project_3.local.attributions.integrated_gradients.mnist_CNN import plot_original_and_attributions_integrated_grad
from project_3.models import get_model


def get_sample_data_cifar(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dm = CIFAR10DataModule()
    dm.setup('')
    dataloader = dm.test_dataloader()

    images, labels = next(iter(dataloader))

    return images[index], labels[index]


def plot_original_and_attributions_with_slic2(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().numpy()

    # Apply SLIC segmentation
    segments = slic(original_img, n_segments=100, compactness=10, sigma=1)
    slic_image = label2rgb(segments, original_img, kind='overlay')

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the SLIC segmented image
    axes[0].imshow(slic_image)
    axes[0].set_title('SLIC Segmented Image')
    axes[0].axis('off')

    # Plot the original image
    axes[1].imshow(original_img)
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    print(attributions_img.dtype)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    attributions_img = scaler.fit_transform(attributions_img)
    print(f"Attributions min: {attributions_img.min()}")
    print(f"Attributions max: {attributions_img.max()}")
    # Plot the attributions image
    axes[2].imshow(attributions_img, cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_title('Integrated Gradients Attributions')
    axes[2].axis('off')

    plt.tight_layout(pad=2.0)
    plt.show()

def plot_original_and_attributions_with_slic(original: torch.Tensor, attributions: torch.Tensor):
    assert original.dim() == 3, "Original tensor does not have 3 dimensions (channel, height, width)"
    assert attributions.dim() == 3, "Attributions tensor does not have 3 dimensions (channel, height, width)"

    original_img: np.ndarray = original.permute(1, 2, 0).detach().numpy()
    attributions_img: np.ndarray = attributions.permute(1, 2, 0).detach().numpy()

    # Apply SLIC on the attributions image
    segments = slic(original_img, n_segments=100, compactness=10, sigma=1)
    slic_image = label2rgb(segments, original_img, kind='overlay')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # SLIC segmentation
    ax1.imshow(slic_image)
    ax1.set_axis_off()
    ax1.set_title('SLIC segmented image')

    # Original image
    ax2.imshow(original_img)
    ax2.set_axis_off()
    ax2.set_title('Original image')

    # Overlayed Integrated Gradients
    viz.visualize_image_attr(attributions_img,
                             original_image=original_img,
                             method='blended_heat_map',
                             sign="all",
                             cmap="RdBu",
                             title="Overlayed Integrated Gradients",
                             plt_fig_axis=(fig, ax3))

    fig.tight_layout()
    plt.show()

def _main():
    model: CifarLargeModel = get_model("kuba_cifar_large.pt")

    for i in range(100):
        img, label = get_sample_data_cifar(i)
        img.requires_grad = True

        ig = IntegratedGradients(model)
        model.eval()
        model.zero_grad()
        gradients = ig.attribute(img.unsqueeze(0), target=label.item())

        # Predict the class
        predicted_label = model(img.unsqueeze(0)).argmax().item()
        print(f"True label: {label.item()}, Predicted label: {predicted_label}")

        # plot_original_and_attributions_integrated_grad(img, gradients)
        plot_original_and_attributions_with_slic(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
