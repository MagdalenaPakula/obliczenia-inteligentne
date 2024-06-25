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


def lbp_feature_extraction(image):
    image = image.squeeze().numpy()
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    return torch.from_numpy(lbp)


def _main():
    image, label = get_transformed_data(index=0)

    # Plot the LBP image
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title(f'LBP Image - Label: {label}')
    plt.show()

if __name__ == '__main__':
    _main()