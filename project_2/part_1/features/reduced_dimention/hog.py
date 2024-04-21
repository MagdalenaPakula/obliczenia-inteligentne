import torch
from skimage.feature import hog

from project_2.part_1.data.MNIST import load_dataset_MNIST


def hog_feature_extraction(image):
    image_np = image.numpy()

    hog_features, hog_image = hog(image_np, orientations=9, pixels_per_cell=(14, 14),
                                  cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')

    hog_features_tensor = torch.tensor(hog_features, dtype=torch.float32)

    return hog_features_tensor


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    image = dataset_mnist[0][0][0]
    print("\nReduced-dimension feature extraction: Histogram of Oriented Gradients (HOG):")
    print(hog_feature_extraction(image))

