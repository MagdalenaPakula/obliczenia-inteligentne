from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def load_other_datasets():
    # Load other datasets (replace with your actual loading code)
    iris_data = load_iris()
    wine_data = load_wine()
    breast_cancer_data = load_breast_cancer()

    # Fit scaler to each dataset and transform them
    scaled_data = [StandardScaler().fit_transform(dataset.data) for dataset in
                   [iris_data, wine_data, breast_cancer_data]]

    # Split data into train and test sets (assuming separate target variable exists)
    train_data, test_data = [], []
    for data, target in zip(scaled_data, [iris_data.target, wine_data.target, breast_cancer_data.target]):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
        train_data.append(CustomDataset(X_train, y_train))
        test_data.append(CustomDataset(X_test, y_test))

    return [{'train_dataset': train_data[i], 'test_dataset': test_data[i], 'name': name} for i, name in
            enumerate(['Iris Dataset', 'Wine Dataset', 'Breast Cancer Wisconsin Dataset'])]
