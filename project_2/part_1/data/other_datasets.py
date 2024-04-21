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
        sample = {'data': self.data[idx], 'target': self.target[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_other_datasets():
    # Load other datasets
    iris_data = load_iris()
    wine_data = load_wine()
    breast_cancer_data = load_breast_cancer()

    # Create StandardScaler instances for each dataset
    scalers = [StandardScaler() for _ in range(3)]

    # Fit scaler to each dataset and transform them
    scaled_data = [scalers[i].fit_transform(dataset.data) for i, dataset in
                   enumerate([iris_data, wine_data, breast_cancer_data])]

    # Split data into train and test sets
    train_data, test_data = [], []
    for data, dataset in zip(scaled_data, [iris_data, wine_data, breast_cancer_data]):
        X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.2, random_state=42)
        train_data.append(CustomDataset(X_train, y_train))
        test_data.append(CustomDataset(X_test, y_test))

    return [{'train_dataset': train_data[i], 'test_dataset': test_data[i], 'name': name} for i, name in
            enumerate(['Iris Dataset', 'Wine Dataset', 'Breast Cancer Wisconsin Dataset'])]

