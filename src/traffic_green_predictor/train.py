import torch
from torch.utils.data import DataLoader
from traffic_green_predictor.dataset import TrafficDataset
from traffic_green_predictor.model import TrafficModel
import torch.optim as optim
import torch.nn as nn

data_maps = {
    "K1": "data/4505_k1.csv",
    "K2": "data/8705_k2.csv",
    "K3": "data/4505_k3.csv",
    "K4": "data/8705_k4.csv"
}

def train(signal_name):
    dataset = TrafficDataset(data_maps[signal_name])
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TrafficModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    file_name = f"models/model_{signal_name}.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")

if __name__ == "__main__":
    for signal in ["K1", "K2", "K3", "K4"]:
        print(f"Training model for {signal}")
        train(signal)
