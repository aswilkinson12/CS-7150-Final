import os
import numpy as np
import torch
from torch.utils.data import Dataset

# loads past and future trajectory pairs for trajectory prediction
class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split

        # paths
        past_path = os.path.join(root_dir, f"{split}_past.npy")
        future_path = os.path.join(root_dir, f"{split}_future.npy")

        # load np arrays
        self.past = np.load(past_path)      #  (N, 4, 2)
        self.future = np.load(future_path)  #  (N, 6, 2)

        assert len(self.past) == len(self.future), "past & future count mismatch"

    def __len__(self):
        return len(self.past)

    #  past: (PAST_STEPS, 2) future: (FUTURE_STEPS, 2)
    def __getitem__(self, idx):
        past = torch.tensor(self.past[idx], dtype=torch.float32)
        future = torch.tensor(self.future[idx], dtype=torch.float32)
        return past, future


# debug test 
if __name__ == "__main__":
    dataset = TrajectoryDataset(root_dir="data", split="train")
    print("Dataset size:", len(dataset))
    past, future = dataset[0]
    print("Past sample shape:", past.shape)      # should be torch.Size([4,2])
    print("Future sample shape:", future.shape)  # shoule be torch.Size([6,2])
