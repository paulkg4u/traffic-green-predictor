import torch
from traffic_green_predictor.model import TrafficModel
import numpy as np


def predict_one(signal_name,hour, minute, dow):
    model = TrafficModel()
    model.load_state_dict(torch.load(f"models/model_{signal_name}.pth"))
    model.eval()

    x = torch.tensor([[hour, minute, dow]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).numpy().flatten()

    print(f"Raw predictions: {pred}...")  # print first 10
    return (pred > 0.5).astype(int)

if __name__ == "__main__":
    signal_num = int(input("Enter signal number (1-4): "))
    signal_name = f"K{signal_num}"
    hour = int(input("Enter hour (0-23): "))
    minute = int(input("Enter minute (0-59): "))
    dow = 0  # Assume Monday, or could ask for input

    result = predict_one(signal_name, hour, minute, dow)
    print("Prediction:", result)
    print("Green seconds:", np.where(result == 1)[0])
    if np.sum(result) == 0:
        print("No green predicted for this time. Try a different hour/minute.")
