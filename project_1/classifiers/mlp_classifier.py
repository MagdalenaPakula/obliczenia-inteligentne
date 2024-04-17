import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from project_1.data import load_generated_datasets
from project_1.visualization import plot_decision_boundary


def mlp_experiment(X, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-1],
                                                        X[:, -1],
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)

    activations = ['identity', 'relu']  # Different activation functions
    best_mlp = None
    best_mlp_accuracy = 0

    for activation in activations:
        mlp = MLPClassifier(hidden_layer_sizes=(5,),
                            activation=activation,
                            max_iter=100_000,
                            tol=0,
                            n_iter_no_change=100_000,
                            solver='sgd')
        mlp.fit(X_train, y_train)

        accuracy_train = mlp.score(X_train, y_train)
        if accuracy_train > best_mlp_accuracy:
            best_mlp_accuracy = accuracy_train
            best_mlp = mlp

        plot_decision_boundary(lambda x: mlp.predict(x), X[:, :2], mlp.predict(X[:, :2]),
                               f"MLP ({activation}) - {dataset_name}")
        # Visualize decision boundary
        # plot_voronoi_diagram(X_train[:, :2], mlp.predict(X_train[:, :2]), y_train,
        #                      f"MLP ({activation}) - {dataset_name}")

        # Confusion matrix
        cm = confusion_matrix(y_train, mlp.predict(X_train))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix for MLP ({activation}) on {dataset_name}")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.show()

    return best_mlp


if __name__ == '__main__':
    for dataset, name in load_generated_datasets()[4:5]:
        mlp_experiment(dataset, name)
