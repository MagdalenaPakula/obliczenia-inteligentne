from pathlib import Path

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, Dataset

from project_2.part_1.data.MNIST import load_dataset_MNIST

DATA_DIR = Path(__file__).parent / Path('./data/MNIST_PCA/raw/')


def save(X_train, y_train, X_test, y_test):
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    torch.save(X_train, DATA_DIR / 'X_train.pt')
    torch.save(X_test, DATA_DIR / 'X_test.pt')
    torch.save(y_train, DATA_DIR / 'y_train.pt')
    torch.save(y_test, DATA_DIR / 'y_test.pt')


def try_load() -> dict[str, Dataset] | None:
    try:
        X_train = torch.load(DATA_DIR / 'X_train.pt')
        y_train = torch.load(DATA_DIR / 'y_train.pt')
        X_test = torch.load(DATA_DIR / 'X_test.pt')
        y_test = torch.load(DATA_DIR / 'y_test.pt')

        return {
            'train_dataset': TensorDataset(X_train, y_train),
            'test_dataset': TensorDataset(X_test, y_test)
        }

    except FileNotFoundError:
        return None


def reduce_dimensions_pca(features, labels):
    pca = PCA(n_components=2, random_state=42)
    print("Calculating PCA embedding... ", end='')
    transformed_features = pca.fit_transform(features, labels)
    print("Done")

    return transformed_features


def transform_dataset(dataset):
    train_dataset = dataset['train_dataset']
    test_dataset = dataset['test_dataset']

    X_train = train_dataset.data.flatten(1)
    y_train = train_dataset.targets
    X_test = test_dataset.data.flatten(1)
    y_test = test_dataset.targets

    train_count = X_train.size(0)

    X = torch.cat((X_train, X_test), dim=0)
    y = torch.cat((y_train, y_test), dim=0)

    X = torch.tensor(reduce_dimensions_pca(X.numpy(), y.numpy()))

    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X)

    X_train = X[:train_count]
    X_test = X[train_count:]

    # save transformed features
    save(X_train, y_train, X_test, y_test)

    return {
        'name': dataset['name'],
        'train_dataset': TensorDataset(X_train, y_train),
        'test_dataset': TensorDataset(X_test, y_test)
    }


def get_mnist_pca():
    # try to load pre-transformed dataset from disk
    print("Attempting to read dataset from disk... ", end='')
    loaded = try_load()
    if loaded:
        print("Done")
        return {
            'name': 'MNIST',
            'train_dataset': loaded['train_dataset'],
            'test_dataset': loaded['test_dataset']
        }

    print("Failed")
    # load untransformed dataset
    dataset_mnist = load_dataset_MNIST()

    return transform_dataset(dataset_mnist)

def plot_dataset_pca(dataset):
    NUM_POINTS = 1000

    X = dataset['train_dataset'].tensors[0][:NUM_POINTS].numpy()
    X = MinMaxScaler().fit_transform(X)
    y = dataset['train_dataset'].tensors[1][:NUM_POINTS].numpy()

    for digit in range(10):
        plt.scatter(X[y == digit, 0], X[y == digit, 1], color=plt.cm.Dark2(digit), marker=f'${digit}$')


if __name__ == "__main__":
    new_dataset = get_mnist_pca()

    plot_dataset_pca(new_dataset)
    plt.show()