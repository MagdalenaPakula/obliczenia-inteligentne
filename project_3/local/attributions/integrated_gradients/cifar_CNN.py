import torch
from captum.attr import IntegratedGradients
from typing import Tuple
from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models.kuba.cifar.large import CifarLargeModel, _CIFARLargeFeatureExtractor
from project_3.local.attributions.integrated_gradients.mnist_CNN import plot_original_and_attributions_integrated_grad
from project_3.models import get_model


def get_sample_data_cifar(index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    dm = CIFAR10DataModule()
    dm.setup('')
    dataloader = dm.test_dataloader()

    images, labels = next(iter(dataloader))

    return images[index], labels[index]


def _main():
    model: CifarLargeModel = get_model("kuba_cifar_large.pt")

    for i in range(10):
        img, label = get_sample_data_cifar(i)
        img.requires_grad = True

        ig = IntegratedGradients(model)
        model.eval()
        model.zero_grad()
        gradients = ig.attribute(img.unsqueeze(0), target=label.item())

        # Predict the class
        predicted_label = model(img.unsqueeze(0)).argmax().item()
        print(f"True label: {label.item()}, Predicted label: {predicted_label}")

        plot_original_and_attributions_integrated_grad(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
