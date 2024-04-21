import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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


def transform_MNIST_2d(image):
    # Reshape the image to a 2D tensor (28x28) - 1 WAY
    return image.reshape(28, 28).clone().detach()


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

    # Extract features from MNIST dataset 2D
    mnist_features_flat_train = CustomMNISTDataset(datasets_mnist[0][0], datasets_mnist[0][1],
                                                   transform=transform_MNIST_flat)
    mnist_features_flat_test = CustomMNISTDataset(datasets_mnist[1][0], datasets_mnist[1][1],
                                                  transform=transform_MNIST_flat)

    mnist_features_2d_train = CustomMNISTDataset(datasets_mnist[0][0], datasets_mnist[0][1],
                                                 transform=transform_MNIST_2d)
    mnist_features_2d_test = CustomMNISTDataset(datasets_mnist[1][0], datasets_mnist[1][1],
                                                transform=transform_MNIST_2d)

    # Visualize feature extraction
    visualize_feature_extraction(mnist_features_2d_train, "MNIST Features (2D) Train")
    visualize_feature_extraction(mnist_features_2d_test, "MNIST Features (2D) Test")
