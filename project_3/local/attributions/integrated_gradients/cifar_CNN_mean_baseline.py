from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from skimage.color import label2rgb
from skimage.segmentation import slic

from project_2.part_2.data import CIFAR10DataModule
from project_2.part_2.models.kuba.cifar.large import CifarLargeModel, _CIFARLargeFeatureExtractor
from project_3.local.attributions.integrated_gradients.cifar_CNN import get_sample_data_cifar
from project_3.local.attributions.integrated_gradients.mnist_CNN import plot_original_and_attributions_integrated_grad
from project_3.models import get_model

def compute_mean_baseline(dm) -> torch.Tensor:
    dataloader = dm.train_dataloader()
    total_sum = torch.zeros((3, 32, 32))
    total_count = 0

    for images, _ in dataloader:
        total_sum += images.sum(dim=0)
        total_count += images.size(0)

    mean_image = total_sum / total_count
    return mean_image


def _main():
    model: CifarLargeModel = get_model("kuba_cifar_large.pt")
    dm = CIFAR10DataModule()
    dm.setup('')

    mean_baseline = compute_mean_baseline(dm)

    for i in range(100):
        img, label = get_sample_data_cifar(i)
        img.requires_grad = True

        ig = IntegratedGradients(model)
        model.eval()
        model.zero_grad()
        gradients = ig.attribute(img.unsqueeze(0), target=label.item(), baselines=mean_baseline.unsqueeze(0))

        # Predict the class
        predicted_label = model(img.unsqueeze(0)).argmax().item()
        print(f"True label: {label.item()}, Predicted label: {predicted_label}")

        plot_original_and_attributions_integrated_grad(img, gradients.squeeze(0))
        #plot_original_and_attributions_with_slic(img, gradients.squeeze(0))


if __name__ == '__main__':
    _main()
