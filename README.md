# Traffic Green Predictor

A PyTorch-based machine learning project to predict traffic light green periods for specific signals. The model predicts whether each second in a minute will be green based on the hour, minute, and day of the week.

## Features

- Predicts green status for 60 seconds in a minute
- Uses LSTM-based neural network for temporal continuity
- Supports multiple traffic signals (K1, K2, K3, K4)
- Trained on historical traffic data

## Installation

1. Ensure you have Python 3.12+ installed
2. Install uv package manager if not already installed
3. Clone the repository and navigate to the project directory
4. Install dependencies:
   ```bash
   uv sync
   uv pip install -e .
   ```

## Usage

### Training

Train the model for a specific signal:

```bash
uv run python src/traffic_green_predictor/train.py
```

This trains the model on data for signal K1 and saves it to `models/model_K1.pth`.

### Prediction

Predict green periods for a specific time:

```bash
uv run python src/traffic_green_predictor/predict.py
```

This loads the trained model and predicts for hour=14, minute=37, day_of_week=0, outputting the binary array and green seconds.

## Model Architecture

The model uses:
- Encoder: Linear layer to encode input features (hour, minute, dow)
- LSTM: To learn temporal dependencies in the 60-second sequence
- Decoder: Linear layer to output probabilities for each second

This architecture ensures that predicted green periods are continuous rather than scattered.

## Data

The project uses CSV files with traffic signal data:
- `data/4505_k1.csv`, `data/4505_k3.csv` for signals K1 and K3
- `data/8705_k2.csv`, `data/8705_k4.csv` for signals K2 and K4

Each CSV contains timestamped data with green status (1/0) for each second.

## Dependencies

- PyTorch
- Pandas
- NumPy
- Matplotlib

## License

Add your license here