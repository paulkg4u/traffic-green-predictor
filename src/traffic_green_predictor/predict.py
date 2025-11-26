import torch
from traffic_green_predictor.model import TrafficModel
import numpy as np


data_maps = {
    "K1": "data/4505_k1.csv",
    "K2": "data/4505_k2.csv",
    "K3": "data/4505_k3.csv",
    "K4": "data/4505_k4.csv"
}

def predict_one(signal_name,hour, minute, dow):
    model = TrafficModel()
    model.load_state_dict(torch.load(f"models/model_{signal_name}.pth"))
    model.eval()

    x = torch.tensor([[hour, minute, dow]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).numpy().flatten()

    return (pred > 0.5).astype(int)

if __name__ == "__main__":
    result = predict_one("K1",12, 37, 0)
    print(result)
    print("Green seconds:", np.where(result == 1)[0])
