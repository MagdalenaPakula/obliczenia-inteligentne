import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from skimage.feature import local_binary_pattern, hog

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class CustomMNISTDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.target[idx]
        if self.transform:
            image = self.transform(image)
        return image, target


def transform_MNIST_flat(image):
    # Flatten the image to a 1D tensor with 784 elements
    return torch.flatten(image)


def load_dataset_MNIST():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return [
        (mnist_train.data, mnist_train.targets, 'MNIST Train Dataset'),
        (mnist_test.data, mnist_test.targets, 'MNIST Test Dataset')
    ]


def visualize_MNIST(data):
    images, targets, _ = data
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(targets[i].item())
    plt.show()


# Method 1: Two-dimensional feature extraction for the MNIST dataset using MEAN and STD
def two_dimensional_feature_extraction(image):
    image = image.float()
    mean = image.mean()
    std = image.std()
    return torch.tensor([mean, std], dtype=torch.float32)


# Method 2: Two-dimensional feature extraction for the MNIST dataset using MEAN and VARIANCE
def central_moments_feature_extraction(image):
    image = image.float()
    mean = image.mean()
    variance = image.var()
    return torch.tensor([mean, variance], dtype=torch.float32)


# Method 3: Two-dimensional feature extraction for the MNIST dataset using PCA
def pca_feature_extraction(image, n_components=2):
    # Flatten images (convert from 2D to 1D)
    flattened_image = image.view(image.shape[0], -1)

    # Convert to numpy array for PCA (PyTorch tensors not directly supported)
    data_np = flattened_image.detach().cpu().numpy()

    # Apply PCA with specified number of components
    pca = PCA(n_components=n_components)
    pca.fit(data_np)  # Fit the PCA model

    # Extract information about PCA
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_

    # Print PCA information
    print(f"PCA Explained Variance Ratio: {explained_variance_ratio}")
    print(f"PCA Components: {components}")

    # Transform data using fitted PCA
    pca_features = pca.transform(data_np)

    # Convert features back to torch tensor
    pca_features_tensor = torch.from_numpy(pca_features)

    return pca_features_tensor


# Method 1: Reduced-dimension feature extraction using Histogram of Oriented Gradients (HOG)
def hog_feature_extraction(image):
    # Convert image to numpy array
    image_np = image.numpy()

    # Calculate HOG features
    hog_features, hog_image = hog(image_np, orientations=9, pixels_per_cell=(14, 14),
                                  cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')

    # Convert HOG features to tensor
    hog_features_tensor = torch.tensor(hog_features, dtype=torch.float32)

    return hog_features_tensor


# Method 2: Reduced-dimension feature extraction using Local Binary Patterns (LBP)
def lbp_feature_extraction(image, num_points=8, radius=1, method='uniform'):
    # Convert image to numpy array
    image_np = image.numpy()

    # Calculate LBP features
    lbp = local_binary_pattern(image_np, num_points, radius, method)

    # Calculate histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

    # Convert histogram to tensor
    lbp_features_tensor = torch.tensor(hist, dtype=torch.float32)

    return lbp_features_tensor


if __name__ == "__main__":
    # Load MNIST datasets
    datasets_mnist = load_dataset_MNIST()

    # Visualize MNIST dataset
    visualize_MNIST(datasets_mnist[0])

    # Extract features from the first image using both feature extraction methods
    image, _ = datasets_mnist[0][0][0], datasets_mnist[0][1][0]
    print("Two-dimensional feature extraction results:")
    print(two_dimensional_feature_extraction(image))

    print("\nTwo-dimensional feature extraction results using Central Moments:")
    print(central_moments_feature_extraction(image))

    print("\nTwo-dimensional feature extraction results using PCA:")
    print(pca_feature_extraction(image))

    print("\nReduced-dimension feature extraction: Histogram of Oriented Gradients (HOG):")
    print(hog_feature_extraction(image))

    print("\nReduced-dimension feature extraction: Local Binary Patterns (LBP):")
    print(lbp_feature_extraction(image))