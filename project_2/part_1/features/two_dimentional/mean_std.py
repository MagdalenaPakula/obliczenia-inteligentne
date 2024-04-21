import torch

from project_2.part_1.data.MNIST import load_dataset_MNIST


def mean_std_feature_extraction(image):
    # Calculate mean and standard deviation for each pixel
    image = image.float()
    mean = torch.mean(image, dim=(1, 2))
    std = torch.std(image, dim=(1, 2))

    # Concatenate mean and standard deviation
    mean_std_features = torch.stack((mean, std), dim=1)

    return mean_std_features


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    # Extract mean and standard deviation features from the first image
    image = dataset_mnist[0][0][0]
    print("\nMean and Standard Deviation feature extraction results:")
    print(mean_std_feature_extraction(image.unsqueeze(0)))
