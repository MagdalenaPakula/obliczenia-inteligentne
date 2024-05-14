import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

from project_2.part_1.data.MNIST import load_dataset_MNIST


def pca_features(data, n_components=2):
    flattened_data = data.reshape(-1, data.shape[1] * data.shape[2])

    # Create a PCA object
    pca = PCA(n_components=n_components)

    # Fit the PCA to the data
    pca.fit(flattened_data)

    # Transform the data using the fitted PCA
    transformed_data = pca.transform(flattened_data)

    return transformed_data


if __name__ == "__main__":
    datasets_mnist = load_dataset_MNIST()
    train_images = datasets_mnist['train_dataset'].data
    train_targets = datasets_mnist['train_dataset'].targets

    n_components = 2

    pca_train_features = pca_features(train_images, n_components=n_components)

    # Convert PCA features and targets to a DataFrame for Seaborn
    pca_df = pd.DataFrame(data=pca_train_features, columns=[f'Principal Component {i + 1}' for i in range(n_components)])
    pca_df['label'] = train_targets.numpy()

    # Visualize the distribution of the PCA features
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='label', palette='tab10', data=pca_df, legend='full', alpha=0.6)
    plt.title('PCA of MNIST Dataset: First Two Principal Components')
    plt.show()
    # emphasizing the first two principal components that capture the most variance in the data. The plot shows how the digit classes are distributed in this reduced-dimensional space, highlighting the variance captured by the first two principal components.

    # Print the first three sample digits and their corresponding PCA features
    for i in range(3):
        # Print digit image
        plt.figure()
        plt.imshow(train_images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"Digit: {train_targets[i].item()}")
        plt.axis('off')

        # Print PCA features
        pca_feat_str = f"PCA features: [{pca_train_features[i, 0]:.2f}, {pca_train_features[i, 1]:.2f}]"
        plt.text(0, 30, pca_feat_str, fontsize=12, color='black')

        plt.show()




    # print("First 3 sample digits and their corresponding PCA features:")
    # for i in range(3):
    #     # plt.figure()
    #     # plt.imshow(train_images[i].numpy().squeeze(), cmap='gray')
    #     # plt.title(f"Digit: {train_targets[i].item()}")
    #     # plt.axis('off')
    #     # plt.show()
    #     print(f"PCA features for digit {train_targets[i].item()}:")
    #     print(pca_train_features[i, :2])



