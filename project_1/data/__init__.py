import os

import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler


# Load generated datasets: 1_1.csv, 1_2.csv, 1_3.csv, 2_1.csv, 2_2.csv, 2_3.csv
def load_generated_datasets():
    # get path to data directory as the directory of this file
    data_dir = os.path.dirname(os.path.abspath(__file__))
    # Load files
    data_files = ['1_1.csv', '1_2.csv', '1_3.csv', '2_1.csv',
                  '2_2.csv', '2_3.csv']
    data_sets = []
    for file in data_files:
        df = pd.read_csv(os.path.join(data_dir, file), header=None, delimiter=';')
        data_sets.append((StandardScaler().fit_transform(df.values), file))

    return data_sets


# Load datasets: Iris, Wine and Breast_cancer
def load_other_datasets():
    # Load other datasets
    iris_data = load_iris()
    wine_data = load_wine()
    breast_cancer_data = load_breast_cancer()

    # Create StandardScaler instances for each dataset
    scaler_iris = StandardScaler()
    scaler_wine = StandardScaler()
    scaler_breast_cancer = StandardScaler()

    # Fit scaler to each dataset and transform them
    scaled_iris_data = scaler_iris.fit_transform(iris_data.data)
    scaled_wine_data = scaler_wine.fit_transform(wine_data.data)
    scaled_breast_cancer_data = scaler_breast_cancer.fit_transform(breast_cancer_data.data)

    return [
        (scaled_iris_data, 'Iris Dataset'),
        (scaled_wine_data, 'Wine Dataset'),
        (scaled_breast_cancer_data, 'Breast Cancer Wisconsin Dataset')
    ]
