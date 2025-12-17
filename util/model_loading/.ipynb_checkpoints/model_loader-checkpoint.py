import torch

MODEL_PATH = "../best_efficientnet_b4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path=MODEL_PATH, device=DEVICE):
    return torch.load(path, map_location=device)