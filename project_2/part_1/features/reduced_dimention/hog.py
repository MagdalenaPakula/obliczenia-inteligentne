import matplotlib.pyplot as plt
import torch
from skimage.feature import hog

from project_2.part_1.data.MNIST import load_dataset_MNIST


def hog_feature_extraction(image: torch.Tensor, visualize: bool = False) -> torch.Tensor:
    image_np = image.squeeze().numpy()

    hog_features, hog_image = hog(image_np, orientations=9, pixels_per_cell=(14, 14),
                                  cells_per_block=(2, 2), visualize=True, block_norm='L1')

    hog_features_tensor: torch.Tensor = torch.tensor(hog_features, dtype=torch.float32)

    if visualize:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_np, cmap='gray')
        ax[0].title.set_text('Original Image')
        ax[1].imshow(hog_image)
        ax[1].title.set_text('Hog Features')
        plt.show()
    assert hog_features_tensor.ndimension() == 1, "Extracted features has to be 1D tensor"
    return hog_features_tensor


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    i = 0
    visited = set()
    while len(visited) < 10:
        image = dataset_mnist[0][0][i]
        label = dataset_mnist[0][1][i].item()
        i += 1
        if label in visited:
            continue
        visited.add(label)

        print("\nReduced-dimension feature extraction: Histogram of Oriented Gradients (HOG):")
        feature_extraction = hog_feature_extraction(image, visualize=True)
        print(feature_extraction)
        print(feature_extraction.shape)
