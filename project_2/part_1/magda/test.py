import os

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import datasets, transforms

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'target': self.target[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_other_datasets():
    # Load other datasets
    iris_data = load_iris()
    wine_data = load_wine()
    breast_cancer_data = load_breast_cancer()

    # Create StandardScaler instances for each dataset
    scaler_iris = StandardScaler()
    scaler_wine = StandardScaler()
    scaler_breast_cancer = StandardScaler()

    # Fit scaler to each dataset and transform them
    scaled_iris_data = scaler_iris.fit_transform(iris_data.data)
    scaled_wine_data = scaler_wine.fit_transform(wine_data.data)
    scaled_breast_cancer_data = scaler_breast_cancer.fit_transform(breast_cancer_data.data)

    # Create CustomDataset instances for each dataset
    iris_dataset = CustomDataset(scaled_iris_data, iris_data.target)
    wine_dataset = CustomDataset(scaled_wine_data, wine_data.target)
    breast_cancer_dataset = CustomDataset(scaled_breast_cancer_data, breast_cancer_data.target)

    return [
        {'dataset': iris_dataset, 'name': 'Iris Dataset'},
        {'dataset': wine_dataset, 'name': 'Wine Dataset'},
        {'dataset': breast_cancer_dataset, 'name': 'Breast Cancer Wisconsin Dataset'}
    ]


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


def transform_MNIST_2d(image):
    # Reshape the image to a 2D tensor (28x28)
    return image.reshape(28, 28).clone().detach()


def transform_MNIST_small(image):
    # Convert the image to numpy array
    image_np = image.numpy()

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=28)  # You can adjust the number of components as needed
    image_pca = pca.fit_transform(image_np)

    # Convert back to tensor
    image_tensor = torch.tensor(image_pca, dtype=torch.float32)

    return image_tensor


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
    import matplotlib.pyplot as plt

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


def visualize_feature_extraction(data, title):
    features, targets = zip(*data)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(features[i], cmap=plt.cm.binary)
        plt.xlabel(targets[i].item())
    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    # Load MNIST datasets
    datasets_mnist = load_dataset_MNIST()

    # Visualize MNIST dataset
    visualize_MNIST(datasets_mnist[0])

    # Extract features from MNIST dataset
    mnist_features_flat_train = CustomMNISTDataset(datasets_mnist[0][0], datasets_mnist[0][1], transform=transform_MNIST_flat)
    mnist_features_2d_train = CustomMNISTDataset(datasets_mnist[0][0], datasets_mnist[0][1], transform=transform_MNIST_2d)
    mnist_features_small_train = CustomMNISTDataset(datasets_mnist[0][0], datasets_mnist[0][1], transform=transform_MNIST_small)

    mnist_features_flat_test = CustomMNISTDataset(datasets_mnist[1][0], datasets_mnist[1][1], transform=transform_MNIST_flat)
    mnist_features_2d_test = CustomMNISTDataset(datasets_mnist[1][0], datasets_mnist[1][1], transform=transform_MNIST_2d)
    mnist_features_small_test = CustomMNISTDataset(datasets_mnist[1][0], datasets_mnist[1][1], transform=transform_MNIST_small)

    # Visualize feature extraction
    visualize_feature_extraction(mnist_features_2d_train, "MNIST Features (2D) Train")
    visualize_feature_extraction(mnist_features_small_train, "MNIST Features (Small) Train")
    visualize_feature_extraction(mnist_features_2d_test, "MNIST Features (2D) Test")
    visualize_feature_extraction(mnist_features_small_test, "MNIST Features (Small) Test")