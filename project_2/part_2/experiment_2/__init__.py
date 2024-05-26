import logging
import warnings
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar


def _test_subset_size(module_factory: Callable[[int], pl.LightningDataModule],
                      model_factory: Callable[[], pl.LightningModule],
                      subset_size: int,
                      epochs: int) -> (float, float):
    test_accuracies = np.array([])

    for rnd_seed in range(10):
        print(f"Subset size: {subset_size:4} | Model {rnd_seed + 1:2}/10")

        torch.manual_seed(rnd_seed)

        augmented_mnist = module_factory(subset_size)

        trainer = pl.Trainer(max_epochs=epochs,
                             enable_model_summary=False,
                             enable_progress_bar=True,
                             logger=False,
                             log_every_n_steps=1,
                             callbacks=[TQDMProgressBar(refresh_rate=1)]
                             )
        model = model_factory()
        trainer.fit(model, augmented_mnist)

        test_data = trainer.test(model, augmented_mnist, verbose=False)
        test_accuracy = test_data[0]["test_acc"]

        test_accuracies = np.append(test_accuracies, test_accuracy)

    mean_accuracy = np.mean(test_accuracies)
    std_dev_accuracy = np.std(test_accuracies)

    print(f"===== Subset size: {subset_size} =====")
    print(f"Accuracies: {test_accuracies}")
    print(f"\u03bc = {mean_accuracy:.3f}, \u03c3 = {std_dev_accuracy:.3f} ")
    return mean_accuracy, std_dev_accuracy


def perform_experiment_2(module_factory: Callable[[int], pl.LightningDataModule],
                         model_factory: Callable[[], pl.LightningModule],
                         epochs: int,
                         verbose: bool = False):
    if not verbose:
        warnings.filterwarnings("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    results = dict()

    for subset_size in [10, 200, 1000]:
        mean, std_dev = _test_subset_size(module_factory, model_factory, subset_size, epochs)
        results[subset_size] = (mean, std_dev)

    print(results)
