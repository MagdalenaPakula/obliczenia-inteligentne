import numpy as np
import torch

from project_2.part_1.data.MNIST import load_dataset_MNIST


def histogram_feature_extraction(image):
    # Spłaszczamy obraz do 1D
    image_flat = image.flatten()

    # Obliczamy histogram wartości pikseli
    histogram, _ = np.histogram(image_flat, bins=256)

    # Normalizujemy histogram
    normalized_histogram = histogram / np.sum(histogram)

    # Zwracamy wektor 2-elementowy zawierający znormalizowany histogram
    return normalized_histogram[:2]


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    image = dataset_mnist[0][0][0]
    print("\nHistogram feature extraction results:")
    print(histogram_feature_extraction(image.unsqueeze(0)))

