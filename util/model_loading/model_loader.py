import torch


def load_model(path, device):
    return torch.load(path, map_location=device)