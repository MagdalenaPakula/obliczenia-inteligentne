from captum.robust import MinParamPerturbation
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from project_2.part_2.models.kuba import MnistLargeModel
from project_3.local.attributions.saliency.mnist import get_sample_data
from project_3.models import get_model


def gaussian_noise_attack(image, std=0.1):
    noise = torch.randn_like(image) * std
    perturbed_image = image + noise
    return perturbed_image.clamp(0, 1)


def random_affine_attack(image, degrees):
    transform = transforms.Compose([transforms.RandomAffine(degrees)])
    perturbed_image = transform(image)
    return perturbed_image


def elastic_transform_attack(image, alpha):
    transform = transforms.Compose([transforms.ElasticTransform(alpha=alpha, sigma=5.0)])
    perturbed_image = transform(image)
    return perturbed_image


def cover_top(image, cover_size=10):
    perturbed_image = image.clone()
    perturbed_image[:, :cover_size, :cover_size] = 0.0
    return perturbed_image


def cover_bottom(image, cover_size=10):
    perturbed_image = image.clone()
    perturbed_image[:, :, -cover_size:, :] = 0.0
    return perturbed_image


def plot_original_and_perturbed(original_image, perturbed_image, true_label, predicted_label):
    original_image = original_image.squeeze().detach().cpu().numpy()
    perturbed_image = perturbed_image.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(original_image)
    ax1.set_title(f"Original Image\nTrue Label: {true_label}")
    ax1.axis('off')

    ax2.imshow(perturbed_image)
    ax2.set_title(f"Perturbed Image\nPredicted Label: {predicted_label}")
    ax2.axis('off')

    plt.subplots_adjust(wspace=0.4)

    plt.show()


def _main():
    model: MnistLargeModel = get_model("kuba_mnist_big.pt")

    for i in range(10):
        img, label = get_sample_data(i)
        img.requires_grad = True

        model.eval()
        model.zero_grad()

        min_pert = MinParamPerturbation(
            forward_func=model,
            attack=cover_top,
            arg_name="cover_size",
            arg_min=5,
            arg_max=15,
            arg_step=1,
        )
        perturbed_image, min_cover_size = min_pert.evaluate(img.unsqueeze(0), target=label)
        if perturbed_image is not None:
            print(f"min_cover_size: {min_cover_size}")
            predicted_label = model(perturbed_image).argmax(dim=1).item()
            plot_original_and_perturbed(img, perturbed_image, label, predicted_label)


if __name__ == '__main__':
    _main()
