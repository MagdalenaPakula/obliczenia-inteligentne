from typing import Callable

import pytorch_lightning as pl
import seaborn as sn
import torch
from matplotlib import pyplot as plt

from project_2.part_2.models import saved_models_dir, ModelBase
from project_2.part_2.visualization import plot_decision_boundary


def get_model(model_file: str, trainer: pl.Trainer, data_module: pl.LightningDataModule,
              factory: Callable[[], ModelBase]) -> ModelBase:
    model_path = saved_models_dir / model_file
    try:
        model: ModelBase = torch.load(model_path)
        print("Loaded model from disk")
        return model
    except FileNotFoundError:
        model: ModelBase = factory()
        trainer.fit(model, data_module)
        print("Saving model to disk")
        torch.save(model, model_path)
        return model


def perform_experiment_1(model: ModelBase, model_name: str, trainer: pl.Trainer, data_module: pl.LightningDataModule,
                         show_decision_boundary: bool = False) -> None:
    trainer.test(model, data_module)

    cm = model.confusion_matrix
    fig, ax = plt.subplots()
    sn.heatmap(cm.compute(), annot=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix of {model_name}")
    fig.show()

    if show_decision_boundary:
        print("Plotting decision boundary...")
        plot_decision_boundary(model, data_module.test_dataloader())
        plt.title(f"Decision boundary of {model_name}")
        plt.show()

    print(model)
