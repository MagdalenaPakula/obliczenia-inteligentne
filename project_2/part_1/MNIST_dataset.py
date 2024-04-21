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


if __name__ == "__main__":
    # Load MNIST datasets
    datasets_mnist = load_dataset_MNIST()

    # Visualize MNIST dataset
    visualize_MNIST(datasets_mnist[0])

