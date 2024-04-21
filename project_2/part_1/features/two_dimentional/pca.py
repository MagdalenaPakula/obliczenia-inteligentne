import torch
from sklearn.decomposition import PCA

from project_2.part_1.data.MNIST import load_dataset_MNIST


def pca_feature_extraction(image, n_components=2):
    flattened_image = image.view(image.shape[0], -1)

    data_np = flattened_image.detach().cpu().numpy()

    pca = PCA(n_components=n_components)
    pca.fit(data_np)

    pca_features = pca.transform(data_np)

    pca_features_tensor = torch.from_numpy(pca_features)

    return pca_features_tensor


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    image, _ = dataset_mnist[0][0][0], dataset_mnist[0][1][0]
    print("\nPrincipal Component Analysis (PCA) feature extraction results:")
    print(pca_feature_extraction(image))