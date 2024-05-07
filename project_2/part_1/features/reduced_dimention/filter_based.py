import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2

from project_2.part_1.data.MNIST import load_dataset_MNIST


def univariate_feature_selection(data, target, k=10):
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)
    df['target'] = target

    # Create a UnivariateSelection object using chi-squared test
    selector = SelectKBest(chi2, k=k)

    # Fit the selector to the data and target
    selector.fit(df.iloc[:, :-1], df['target'])

    # Get the selected features
    selected_features = selector.get_support()

    # Create a new DataFrame with only the selected features
    selected_data = df.iloc[:, selected_features]

    return selected_data.to_numpy()


if __name__ == "__main__":
    # Load MNIST dataset
    dataset_mnist = load_dataset_MNIST()

    # Extract features using univariate feature selection
    data, target = dataset_mnist[0][0], dataset_mnist[0][1]
    selected_features = univariate_feature_selection(data, target, k=10)

    print("\nUnivariate Feature Selection (chi-squared) results:")
    print(selected_features.shape)  # Print the shape of the selected features
