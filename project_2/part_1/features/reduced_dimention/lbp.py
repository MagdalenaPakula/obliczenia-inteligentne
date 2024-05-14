import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from project_2.part_1.data.MNIST import load_dataset_MNIST


def lbp_image(image, radius=1, n_points=8):
    image = image.numpy().squeeze()  # Convert from tensor to numpy array
    lbp_img = local_binary_pattern(image, n_points, radius, method='uniform')
    return torch.tensor(lbp_img, dtype=torch.float32).unsqueeze(0)  # Convert back


def lbp_histogram(lbp_image, n_bins=10):
    lbp_image = lbp_image.squeeze().numpy()
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def visualize_LBP(data: torch.utils.data.Dataset):
    plt.figure(figsize=(10, 14))
    for i in range(80):
        image, target = data[i]
        image = image.squeeze().numpy()  # Convert tensor to numpy array for visualization
        plt.subplot(10, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        plt.xlabel(target)
    plt.show()


if __name__ == "__main__":
    # Load MNIST dataset
    datasets_mnist = load_dataset_MNIST(transform=lambda x: lbp_image(x, radius=1, n_points=8))

    # Visualize LBP-transformed MNIST dataset
    print("Visualizing LBP-transformed MNIST dataset:")
    visualize_LBP(datasets_mnist['train_dataset'])

    # Example of computing and visualizing LBP histogram for a single image
    example_image, _ = datasets_mnist['train_dataset'][0]
    lbp_hist = lbp_histogram(example_image, n_bins=10)

    plt.figure()
    plt.bar(range(len(lbp_hist)), lbp_hist, width=0.5, edgecolor='black')
    plt.title('Histogram of LBP codes')
    plt.xlabel('LBP code')
    plt.ylabel('Frequency')
    plt.show()
