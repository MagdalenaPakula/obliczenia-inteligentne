import numpy as np
import torch
from skimage.feature import local_binary_pattern

from project_2.part_1.data.MNIST import load_dataset_MNIST


def lbp_feature_extraction(image, num_points=8, radius=1, method='uniform'):
    image_np = image.numpy()

    # Calculate LBP features
    lbp = local_binary_pattern(image_np, num_points, radius, method)

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

    # Convert histogram to tensor
    lbp_features_tensor = torch.tensor(hist, dtype=torch.float32)

    return lbp_features_tensor


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    image, _ = dataset_mnist[0][0][0], dataset_mnist[0][1][0]
    print("\nLocal Binary Patterns (LBP) feature extraction results:")
    print(lbp_feature_extraction(image))
