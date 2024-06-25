from typing import Tuple

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

from project_2.part_1.data.MNIST import load_dataset_MNIST, visualize_MNIST


def get_transformed_data(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset_MNIST(transform=lbp_feature_extraction)['test_dataset']
    return dataset[index]

def get_original_data(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = load_dataset_MNIST()['test_dataset']
    return dataset[index]


def lbp_feature_extraction(image):
    image = image.squeeze().numpy()
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    return torch.from_numpy(lbp)


def _main():
    # Get the original and LBP-transformed images
    original_image, label = get_original_data(index=0)
    lbp_image, label = get_transformed_data(index=0)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original image
    ax1.imshow(original_image.squeeze(), cmap='gray')
    ax1.set_title(f'Original Image - Label: {label}')
    ax1.set_xticks(np.arange(0, 28, step=4))
    ax1.set_yticks(np.arange(0, 28, step=4))
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Plot the LBP image
    ax2.imshow(lbp_image, cmap='gray')
    ax2.set_title(f'LBP Image - Label: {label}')

    plt.show()


if __name__ == '__main__':
    _main()