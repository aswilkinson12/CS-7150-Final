import torch
import torch.nn as nn

# LSTM baseline model for predicting future positions
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, future_steps=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, future_steps * 2)

    # (B, past_steps, 2) -> (B, future_steps, 2)
    def forward(self, past):
        out, (h_n, c_n) = self.lstm(past) # (B, past_steps, hidden_dim)
        h_last = h_n[-1] # use the last hidden state
        future_flat = self.fc(h_last) # predict all future positions (B, future_steps*2)
        future = future_flat.view(-1, self.future_steps, 2)  # reshape to (B, future_steps, 2)

        return future

# debug test
if __name__ == "__main__":
    model = LSTMModel()
    past = torch.randn(8, 4, 2)   # batch = 8
    future = model(past)
    print("Model output shape:", future.shape)  # (8, 6, 2)
