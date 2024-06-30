import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

from project_2.part_1.data.MNIST import load_dataset_MNIST


def lbp_feature_extraction(image: torch.Tensor, radius: int = 3, n_points: int = 24, method: str = 'uniform',
                           normalize: bool = True, visualize: bool = False) -> torch.Tensor:
    image_np = image.squeeze().numpy()

    lbp_features = local_binary_pattern(image_np, n_points, radius, method=method)

    lbp_features_tensor = torch.tensor(lbp_features.flatten(), dtype=torch.float32)

    if normalize:
        lbp_features_tensor = (lbp_features_tensor - lbp_features_tensor.min()) / (
                    lbp_features_tensor.max() - lbp_features_tensor.min())

    if visualize:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_np, cmap='gray')
        ax[0].title.set_text('Original Image')
        ax[1].imshow(lbp_features, cmap='gray')
        ax[1].title.set_text('LBP Features')
        plt.show()

    assert lbp_features_tensor.ndimension() == 1, "Extracted features has to be 1D tensor"
    return lbp_features_tensor


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    i = 0
    visited = set()
    while len(visited) < 10:
        image = dataset_mnist['train_dataset'].data[i]
        label = dataset_mnist['train_dataset'].targets[i].item()
        i += 1
        if label in visited:
            continue
        visited.add(label)

        print("\nReduced-dimension feature extraction: Local Binary Pattern (LBP):")
        feature_extraction = lbp_feature_extraction(image, visualize=True)
        print(feature_extraction)
        print(feature_extraction.shape)