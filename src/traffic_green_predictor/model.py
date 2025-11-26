import torch
import torch.nn as nn

class TrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.decoder = nn.Linear(128, 1)

    def forward(self, x):
        encoded = self.encoder(x)  # [batch, 64]
        seq = encoded.unsqueeze(1).repeat(1, 60, 1)  # [batch, 60, 64]
        lstm_out, _ = self.lstm(seq)  # [batch, 60, 128]
        out = self.decoder(lstm_out).squeeze(-1)  # [batch, 60]
        return torch.sigmoid(out)
