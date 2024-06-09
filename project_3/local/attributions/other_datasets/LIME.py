from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import numpy as np
import shap

def shap_on_other_datasets(X, y, feature_names, class_names):
    # Scale the input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the MLP model
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_scaled, y)

    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_scaled)

    # Find an example for each class
    for class_name in class_names:
        class_index = np.where(np.array(class_names) == class_name)[0][0]
        class_examples = X_scaled[y == class_index]
        if len(class_examples) > 0:
            instance = class_examples[0]
            prediction = model.predict([instance])[0]
            shap_values = explainer.shap_values([instance])
            print(f"Example for class '{class_name}' is predicted to be {class_names[prediction]}")
            print(f"SHAP values: {shap_values}")
        else:
            print(f"No examples found for class '{class_name}'.")

iris = load_iris()
X, y = iris.data, iris.target
shap_on_other_datasets(X, y, iris.feature_names, iris.target_names)

wine = load_wine()
X, y = wine.data, wine.target
shap_on_other_datasets(X, y, wine.feature_names, wine.target_names)

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
shap_on_other_datasets(X, y, breast_cancer.feature_names, breast_cancer.target_names)
