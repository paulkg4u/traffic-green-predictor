import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df["Time"] = pd.to_datetime(df["Time"])

        df["hour"] = df["Time"].dt.hour
        df["minute"] = df["Time"].dt.minute
        df["second"] = df["Time"].dt.second
        df["dow"] = df["Time"].dt.dayofweek

        grouped = df.groupby(df["Time"].dt.floor("min"))

        X, y = [], []
        for minute, group in grouped:
            sec_vec = group.sort_values("second")["Green Status(1/0)"].tolist()
            if len(sec_vec) == 60:
                X.append([minute.hour, minute.minute, minute.dayofweek])
                y.append(sec_vec)

        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
