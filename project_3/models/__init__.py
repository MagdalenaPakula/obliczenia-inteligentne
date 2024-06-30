from pathlib import Path


import torch


def get_model(filename: str) -> any:
    path = Path(__file__).parent / filename
    return torch.load(path)
