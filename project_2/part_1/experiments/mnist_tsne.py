import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from project_1.visualization import plot_decision_boundary, plot_voronoi_diagram
from project_2.part_1.MLP import MLP
from project_2.part_1.experiments import train_model, evaluate_model
from project_2.part_1.features.two_dimentional.tsne import get_mnist_tsne

MODEL_PATH = "MLP_mnist_tsne.pt"


def get_prediction(model, x: np.ndarray) -> np.ndarray:
    num_predictions = x.shape[0]
    with torch.inference_mode():
        result = model(torch.tensor(x))
        result = result.numpy().astype(np.float32)
        assert result.shape[0] == num_predictions
        return result.argmax(axis=1)


def create_and_train_model(train_loader) -> torch.nn.Module:
    torch.manual_seed(42)
    input_size = 2
    hidden_layer_size = 4
    output_size = 10
    model = MLP(input_size, hidden_layer_size, output_size)
    train_model(model, train_loader, epochs=100, learning_rate=0.05)
    torch.save(model, MODEL_PATH)
    return model


def loader_to_np_array(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    features, labels = np.zeros((0, 2)), np.zeros((0,))
    for data, l in loader:
        features = np.append(features, data, axis=0)
        labels = np.append(labels, l, axis=0)

    return np.array(features), np.array(labels)


def main():
    dataset = get_mnist_tsne()

    name = dataset['name']
    train_dataset = dataset['train_dataset']
    test_dataset = dataset['test_dataset']
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # get model
    try:
        model = torch.load(MODEL_PATH)
    except FileNotFoundError:
        model = create_and_train_model(train_loader)

    evaluate_model(model, test_loader, name)

    x_test, y_fact = loader_to_np_array(test_loader)

    x_test = x_test.astype(np.float32)[:1000]
    y_fact = y_fact.astype(np.float32)[:1000]

    y_pred = get_prediction(model, x_test)

    def classifier(x: np.ndarray) -> np.ndarray:
        return get_prediction(model, x)

    plot_voronoi_diagram(x_test, y_pred, y_fact, colormap=plt.get_cmap('tab10'),
                         diagram_title='MNIST t-SNE voronoi diagram')
    plt.show()

    plot_decision_boundary(classifier, x_test, y_fact, cmap='tab10', resolution=500, size=10,
                           title='MNIST decision boundary with real labels', try_this_if_does_not_work=True)
    plt.show()


if __name__ == '__main__':
    main()
