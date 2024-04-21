import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Turning off warning with TensorFlow's use of oneDNN (formerly known as MKL-DNN) for custom operations
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


def transform_MNIST(image):
    # Flatten the image and reshape/resize it to 28x28
    return torch.tensor(image.flatten().reshape(28, 28), dtype=torch.float32)


def transform_MNIST_small(image):
    # Extract only the first 100 pixels
    return torch.tensor(image[:100], dtype=torch.float32)


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

    return [
        (np.append(scaled_iris_data, iris_data.target.reshape(-1, 1), axis=1), iris_data.target, 'Iris Dataset'),
        (np.append(scaled_wine_data, wine_data.target.reshape(-1, 1), axis=1), wine_data.target, 'Wine Dataset'),
        (np.append(scaled_breast_cancer_data, breast_cancer_data.target.reshape(-1, 1), axis=1),
         breast_cancer_data.target, 'Breast Cancer Wisconsin Dataset')
    ]


def load_dataset_MNIST(transform=transform_MNIST):
    # Load MNIST dataset
    (mnist_data, mnist_target), _ = mnist.load_data()
    mnist_data = mnist_data / 255.0  # Scale pixel values to [0, 1]

    return [
        (mnist_data, mnist_target, 'MNIST Dataset')
    ]


def prepare_data(data, test_size=0.2, random_state=42):
    X, y, dataset_name = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_train, y_train), (X_test, y_test), dataset_name


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
        plt.xlabel(targets[i])
    plt.show()


def extract_features(data, feature_extractor):
    X, y, dataset_name = data
    transformed_data = [(feature_extractor(x), label) for x, label in zip(X, y)]
    return transformed_data, dataset_name


def visualize_feature_extraction(data, title):
    features, targets = zip(*data)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(features[i], cmap=plt.cm.binary)
        plt.xlabel(targets[i])
    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    # Load datasets
    datasets = load_other_datasets() + load_dataset_MNIST(transform=transform_MNIST)

    # Prepare datasets
    for data in datasets:
        (X_train, y_train), (X_test, y_test), dataset_name = prepare_data(data)
        print(f"{dataset_name}:")
        print(f"  Training data shape: {X_train.shape}")
        print(f"  Test data shape: {X_test.shape}")
        print(f"  Training labels shape: {y_train.shape}")
        print(f"  Test labels shape: {y_test.shape}")
        print()

    # Visualize MNIST dataset
    visualize_MNIST(datasets[-1])

    # Extract features from MNIST dataset
    mnist_features_2d, _ = extract_features(datasets[-1], transform_MNIST)
    mnist_features_small, _ = extract_features(datasets[-1], transform_MNIST_small)

    # Visualize feature extraction
    visualize_feature_extraction(mnist_features_2d, "MNIST Features (2D)")
    visualize_feature_extraction(mnist_features_small, "MNIST Features (Small)")