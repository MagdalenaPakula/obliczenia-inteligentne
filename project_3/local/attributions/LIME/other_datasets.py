import numpy as np
from sklearn.neural_network import MLPClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


def lime_on_other_datasets(X, y, feature_names, class_names):
    # Scale the input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the MLP model
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_scaled, y)

    # Explain the model's predictions using LIME
    explainer = LimeTabularExplainer(X_scaled, feature_names=feature_names, class_names=class_names)

    # Find an example for each class
    for class_name in class_names:
        class_index = np.where(np.array(class_names) == class_name)[0][0]
        class_examples = X_scaled[y == class_index]
        if len(class_examples) > 0:
            instance = class_examples[0]
            prediction = model.predict([instance])[0]
            explanation = explainer.explain_instance(instance, model.predict_proba)
            print(f"Example for class '{class_name}' is predicted to be {class_names[prediction]}")
            print(explanation.as_list())
        else:
            print(f"No examples found for class '{class_name}'.")


iris = load_iris()
X, y = iris.data, iris.target
lime_on_other_datasets(X, y, iris.feature_names, iris.target_names)

wine = load_wine()
X, y = wine.data, wine.target
lime_on_other_datasets(X, y, wine.feature_names, wine.target_names)

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
lime_on_other_datasets(X, y, breast_cancer.feature_names, breast_cancer.target_names)
