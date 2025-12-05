import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import TrajectoryDataset
from model import LSTMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
MODEL_PATH = "./model_lstm.pt"

# computes average L2 distance over all steps (B, T, 2)
def ade(pred, gt):
    return torch.norm(pred - gt, dim=2).mean().item()

# computes final-step L2 distance (B, T, 2)
def fde(pred, gt):
    return torch.norm(pred[:, -1] - gt[:, -1], dim=1).mean().item()


def evaluate():
    # load dataset + dataloader
    val_dataset = TrajectoryDataset(DATA_DIR, split="val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # load model
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    total_ade = 0.0
    total_fde = 0.0
    count = 0

    # evaluate
    with torch.no_grad():
        for past, future in val_loader:
            past = past.to(DEVICE)
            future = future.to(DEVICE)

            pred = model(past)

            batch_ade = ade(pred, future)
            batch_fde = fde(pred, future)

            total_ade += batch_ade * len(past)
            total_fde += batch_fde * len(past)
            count += len(past)

    final_ade = total_ade / count
    final_fde = total_fde / count

    print(" Validation Metrics (Baseline LSTM)")
    print(f"ADE: {final_ade:.4f} meters")
    print(f"FDE: {final_fde:.4f} meters")


if __name__ == "__main__":
    evaluate()
