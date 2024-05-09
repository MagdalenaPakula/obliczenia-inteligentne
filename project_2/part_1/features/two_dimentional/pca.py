import torch
from matplotlib import pyplot as plt
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
    images, targets, _ = dataset_mnist[0]

    print("\nPrincipal Component Analysis (PCA) feature extraction results:")
    for i in range(1):
        image, target = images[i], targets[i]
        print("Label:", target.item())
        pca_features = pca_feature_extraction(image)
        print("PCA Features:", pca_features)

