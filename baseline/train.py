import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrajectoryDataset
from model import LSTMModel


# settings
DATA_DIR = "data"
SAVE_PATH = "./model_lstm.pt"

BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # load datasets
    train_dataset = TrajectoryDataset(DATA_DIR, split="train")
    val_dataset   = TrajectoryDataset(DATA_DIR, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model, optimizer, loss
    model = LSTMModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("Training on:", DEVICE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for past, future in train_loader:
            past = past.to(DEVICE)
            future = future.to(DEVICE)

            pred = model(past)
            loss = criterion(pred, future)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for past, future in val_loader:
                past = past.to(DEVICE)
                future = future.to(DEVICE)

                pred = model(past)
                loss = criterion(pred, future)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}  |  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")

    # save model
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
