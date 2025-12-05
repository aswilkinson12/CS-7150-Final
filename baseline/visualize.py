import os
import torch
import matplotlib.pyplot as plt

from dataset import TrajectoryDataset
from model import LSTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
MODEL_PATH = "./model_lstm.pt"

SAVE_DIR = "./plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_trajectory(past, future_gt, future_pred, idx=0):
    plt.figure(figsize=(6,6))
    plt.title(f"Trajectory Visualization #{idx}")

    # past (blue)
    plt.plot(past[:,0], past[:,1], 'bo-', label="Past (input)")

    # ground Truth Future (green)
    plt.plot(future_gt[:,0], future_gt[:,1], 'go-', label="Future GT")

    # predicted Future (red)
    plt.plot(future_pred[:,0], future_pred[:,1], 'ro--', label="Future Pred")

    plt.legend()
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.axis("equal") 
    save_path = os.path.join(SAVE_DIR, f"traj_{idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def visualize(num_samples=5):
    val_dataset = TrajectoryDataset(DATA_DIR, split="val") # load dataset

    # load model
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # visualize 
    for idx in range(min(num_samples, len(val_dataset))):
        past, future_gt = val_dataset[idx]

        past_in = past.unsqueeze(0).to(DEVICE)  # (1, P, 2)
        with torch.no_grad():
            future_pred = model(past_in).cpu().squeeze(0)  # (F,2)

        past = past.numpy()
        future_gt = future_gt.numpy()
        future_pred = future_pred.numpy()

        plot_trajectory(past, future_gt, future_pred, idx=idx)


if __name__ == "__main__":
    visualize(num_samples=10)
