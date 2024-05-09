import os

import matplotlib.pyplot as plt
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


def load_dataset_MNIST(transform=lambda x: x):
    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(transform)
        ]
    )
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return {
        'train_dataset': mnist_train,
        'test_dataset': mnist_test,
        'name': 'MNIST'
    }


def visualize_MNIST(data):
    images, targets, _ = data
    plt.figure(figsize=(2, 2))
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap='gray')
        plt.xlabel(targets[i].item())
    plt.show()


if __name__ == "__main__":
    # Load MNIST datasets
    datasets_mnist = load_dataset_MNIST()

    # Visualize MNIST dataset
    visualize_MNIST(datasets_mnist[0])
