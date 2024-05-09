import torch
from sklearn.manifold import TSNE
from project_2.part_1.data.MNIST import load_dataset_MNIST


def tsne_feature_extraction(image, n_components=2, perplexity=20):
    data_np = image.flatten().detach().cpu().numpy()

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_features = tsne.fit_transform(data_np)

    tsne_features_tensor = torch.from_numpy(tsne_features)

    return tsne_features_tensor


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    image, _ = dataset_mnist[0][0][0], dataset_mnist[0][1][0]
    print("\nt-SNE feature extraction results:")
    print(tsne_feature_extraction(image))
