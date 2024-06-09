from itertools import islice

import lime
import lime.lime_tabular
import numpy as np
import torch
import torch.nn as nn
from lime import lime_image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from project_2.part_1.data.MNIST import load_dataset_MNIST


# Define the model architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the model dimensions
input_size = 28 * 28
hidden_layer_size = 20
output_size = 10

# Instantiate the model with the given dimensions
model = MLP(input_size, hidden_layer_size, output_size)

# Load the state dictionary
MODEL_PATH = "../../models/MLP_mnist_flattened.pt"
state_dict = torch.load(MODEL_PATH)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Load the MNIST dataset
dataset = load_dataset_MNIST(transform=torch.flatten)
test_loader = DataLoader(dataset['test_dataset'], batch_size=1, shuffle=False)

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    np.array(dataset['train_dataset'].data.reshape(-1, 28 * 28)),
    feature_names=[f"pixel_{i}" for i in range(28 * 28)],
    class_names=[str(i) for i in range(10)],
    mode='classification'
)


# Define a function to get model predictions
def predict_fn(data):
    model.eval()
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32)
        logits = model(data).numpy()
    return logits


# Get the first sample from the test set
sample_idx = 0
sample, label = dataset['test_dataset'][sample_idx]
print("Label:", label)

# Convert the sample to a 28x28 image
img = sample.view(28, 28).numpy()
plt.imshow(img, cmap='gray')
plt.show()

# Use LIME to explain the prediction
exp = explainer.explain_instance(
    data_row=sample,
    predict_fn=predict_fn,
    num_features=10  # Number of features to show in explanation
)

print(exp.as_map())

fig = exp.as_pyplot_figure()
plt.show()
