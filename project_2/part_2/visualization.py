import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from project_1.visualization import plot_decision_boundary as _plot_db_impl
from project_2.part_2.models import ModelBase


def _get_prediction(model: nn.Module, x: np.ndarray) -> np.ndarray:
    num_predictions = x.shape[0]
    with torch.inference_mode():
        result = model(torch.tensor(x))
        result = result.numpy().astype(np.float32)
        assert result.shape[0] == num_predictions
        return result.argmax(axis=1)


def _loader_to_np_array(loader: DataLoader, feature_extractor: nn.Module | None = None) \
        -> tuple[np.ndarray, np.ndarray]:
    features, labels = np.zeros((0, 2)), np.zeros((0,))
    for data, l in loader:
        data: torch.Tensor
        l: torch.Tensor
        if feature_extractor is not None:
            data = feature_extractor(data)

        assert data.size(1) == 2, "Feature size should be 2 for plotting"
        features = np.append(features, data.numpy(), axis=0)
        labels = np.append(labels, l.numpy(), axis=0)

    return np.array(features), np.array(labels)


def plot_decision_boundary(model: ModelBase, test_loader: DataLoader) -> None:
    NUM_POINTS = 1000

    def classifier(x):
        return _get_prediction(model.classifier, x)

    model.eval()
    with torch.inference_mode():
        x_test, y_test = _loader_to_np_array(test_loader, model.feature_extractor)
        x_test = x_test[:NUM_POINTS].astype(np.float32)
        y_test = y_test[:NUM_POINTS].astype(np.float32)

        _plot_db_impl(classifier, x_test, y_test,
                      cmap='tab10',
                      size=10,
                      try_this_if_does_not_work=True,
                      resolution=500)
