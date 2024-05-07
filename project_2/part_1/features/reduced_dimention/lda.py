import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from project_2.part_1.data.MNIST import load_dataset_MNIST


def lda_feature_extraction(data, target, n_components=None):
    flattened_data = data.view(data.shape[0], -1)

    allowed_components = min(min(flattened_data.shape), len(set(target)) - 1)

    if n_components is None or n_components > allowed_components:
        n_components = allowed_components
    lda = LDA(n_components=n_components)

    lda.fit(flattened_data, target)

    lda_features = lda.transform(flattened_data)

    return lda_features


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    data, target = dataset_mnist[0][0], dataset_mnist[0][1]
    lda_features = lda_feature_extraction(data, target, n_components=2)

    print("\nLinear Discriminant Analysis (LDA) results:")
    print(lda_features.shape)
