import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from project_2.part_1.data.MNIST import load_dataset_MNIST

def rfe_feature_selection(data, target, k=10, sample_size=1000):
    # Flatten each image
    flattened_data = data.view(data.shape[0], -1)

    # Randomly sample a subset of the data
    np.random.seed(0)  # for reproducibility
    random_indices = np.random.choice(flattened_data.shape[0], sample_size, replace=False)
    sampled_data = flattened_data[random_indices]
    sampled_target = target[random_indices]

    # Create an SVM classifier
    clf = SVC(kernel="linear")

    # Create an RFECV object with 10-fold cross-validation
    rfecv = RFECV(estimator=clf, scoring='accuracy', cv=10)

    # Fit the RFECV to the sampled data and target
    rfecv.fit(sampled_data, sampled_target)

    # Get the selected features
    selected_features = rfecv.support_

    # Create a new data array with only the selected features
    selected_data = sampled_data[:, selected_features]

    return selected_data

if __name__ == "__main__":
    # Load MNIST dataset
    dataset_mnist = load_dataset_MNIST()

    # Extract features using RFE feature selection
    data, target = dataset_mnist[0][0], dataset_mnist[0][1]
    selected_features = rfe_feature_selection(data, target, k=10, sample_size=1000)

    print("\nRFE Feature Selection with SVM results:")
    print(selected_features.shape)  # Print the shape of the selected features
