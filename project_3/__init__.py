import torch
import torchvision.transforms as transforms
from captum.attr import Saliency
from project_2.part_1.MLP import MLP  # Assuming MLP class is defined in MLP.py
from project_2.part_1.data.MNIST import load_dataset_MNIST
from project_2.part_1.features.reduced_dimention.hog import hog_feature_extraction

# Load the saved MLP model
model = torch.load("models/MLP_mnist_hog.pt")
model.eval()

# Define transforms to preprocess input data
preprocess = transforms.Compose([
    hog_feature_extraction,  # Use the same HOG feature extraction method
    transforms.ToTensor(),
])

# Load MNIST dataset with HOG features
dataset = load_dataset_MNIST(transform=preprocess)

# Select a sample input image
input_image, _ = dataset['test_dataset'][0]

# Preprocess the input image
input_image_preprocessed = preprocess(input_image).unsqueeze(0)  # Add batch dimension

# Create a Saliency object
saliency = Saliency(model)

# Compute saliency map
attributions = saliency.attribute(input_image_preprocessed, target=0)  # Assuming target class is 0

# Visualize the attributions
import matplotlib.pyplot as plt

plt.imshow(attributions.squeeze(0).numpy(), cmap='hot', interpolation='nearest')
plt.show()
