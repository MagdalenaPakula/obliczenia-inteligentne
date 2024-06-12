import matplotlib.pyplot as plt

from captum.robust import MinParamPerturbation
from torchvision import transforms

from project_2.part_2.models.kuba.cifar.large import CifarLargeModel, _CIFARLargeFeatureExtractor
from project_3.local.attributions.integrated_gradients.cifar_CNN import get_sample_data_cifar
from project_3.models import get_model

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def random_rotation_attack(image, degrees):
    transform = transforms.Compose([transforms.RandomRotation(degrees)])
    perturbed_image = transform(image)
    return perturbed_image


def plot_original_and_perturbed(original_image, perturbed_image, true_label, predicted_label):
    original_image = original_image.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
    perturbed_image = perturbed_image.squeeze().detach().cpu().numpy().transpose((1, 2, 0))

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
    model: CifarLargeModel = get_model("kuba_cifar_large.pt")

    for i in range(100):
        img, label = get_sample_data_cifar(i)
        img.requires_grad = True

        model.eval()
        model.zero_grad()

        min_pert = MinParamPerturbation(
            forward_func=lambda x: model(x),
            attack=random_rotation_attack,
            arg_name="degrees",
            arg_min=0,
            arg_max=90,
            arg_step=0.01,
        )
        perturbed_image, _ = min_pert.evaluate(img.unsqueeze(0), target=label)
        if perturbed_image is not None:
            print(f"size: {_}")
            predicted_label = model(perturbed_image).argmax(dim=1).item()
            plot_original_and_perturbed(img, perturbed_image, class_names[label.item()], class_names[predicted_label])


if __name__ == '__main__':
    _main()
