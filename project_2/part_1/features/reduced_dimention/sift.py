import cv2
import numpy as np

from project_2.part_1.data.MNIST import load_dataset_MNIST


def sift_feature_extraction(image):
    # Convert image to three channels (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    max_descriptors = 128  # Maximum number of descriptors
    if descriptors is not None:
        if descriptors.shape[0] < max_descriptors:
            descriptors = np.concatenate([descriptors, np.zeros((max_descriptors - descriptors.shape[0], 128))])
        elif descriptors.shape[0] > max_descriptors:
            descriptors = descriptors[:max_descriptors]

    return descriptors.flatten()


if __name__ == "__main__":
    dataset_mnist = load_dataset_MNIST()

    image = dataset_mnist[0][0][0].numpy()  # Convert to numpy
    print("\nSIFT feature extraction results:")
    print(sift_feature_extraction(image))
