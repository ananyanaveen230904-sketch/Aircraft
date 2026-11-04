"""
Train the LSTM RUL model on NASA C-MAPSS FD001.

Run from the project root:
    python src/rul/train.py

Produces:
    models/rul_lstm.pt        (trained PyTorch model weights)
    models/rul_scaler.pkl     (fitted MinMaxScaler, for reuse at inference time)
    output/rul_test_predictions.png   (predicted vs actual RUL scatter, test set)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rul.data_prep import (
    SENSOR_COLUMNS,
    compute_rul_labels,
    create_sequences,
    create_test_sequences_last_window,
    drop_constant_sensors,
    load_cmapss,
    load_rul_labels,
    normalize_sensors,
    save_scaler,
)
from src.rul.model import build_lstm_model, mae_loss, rmse_loss

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cmapss")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")

WINDOW_SIZE = 30
RUL_CEILING = 125
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------------
    # Load + preprocess
    # ---------------------------------------------------------------
    print("\nLoading data...")
    train_df = load_cmapss(os.path.join(DATA_DIR, "train_FD001.txt"))
    test_df = load_cmapss(os.path.join(DATA_DIR, "test_FD001.txt"))
    test_rul_true = load_rul_labels(os.path.join(DATA_DIR, "RUL_FD001.txt"))

    train_df, dropped_sensors = drop_constant_sensors(train_df)
    test_df = test_df.drop(columns=dropped_sensors)
    feature_columns = [c for c in SENSOR_COLUMNS if c not in dropped_sensors]

    train_df, scaler = normalize_sensors(train_df, feature_columns, fit=True)
    test_df, _ = normalize_sensors(test_df, feature_columns, scaler=scaler, fit=False)
    save_scaler(scaler, os.path.join(MODELS_DIR, "rul_scaler.pkl"))

    train_df = compute_rul_labels(train_df, ceiling=RUL_CEILING)

    # ---------------------------------------------------------------
    # Sequences
    # ---------------------------------------------------------------
    print("\nBuilding sequences...")
    X, y = create_sequences(train_df, feature_columns, window_size=WINDOW_SIZE)
    print(f"  Train sequences: X={X.shape}, y={y.shape}")

    X_test, test_engine_ids = create_test_sequences_last_window(test_df, feature_columns, window_size=WINDOW_SIZE)
    # test_rul_true is ordered by engine_id (1..N) per RUL_FD001.txt convention;
    # test_engine_ids follows the same order since groupby sorts by engine_id.
    y_test = np.clip(test_rul_true[: len(test_engine_ids)], 0, RUL_CEILING).astype(np.float32)
    print(f"  Test sequences:  X={X_test.shape}, y={y_test.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=SEED
    )

    def to_loader(X_arr, y_arr, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_arr), torch.from_numpy(y_arr)
        )
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = build_lstm_model(input_shape=(WINDOW_SIZE, X.shape[-1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                val_preds.append(model(xb).cpu())
                val_true.append(yb)
        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)
        val_rmse = rmse_loss(val_preds, val_true).item()
        val_mae = mae_loss(val_preds, val_true).item()

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train_mse={np.mean(train_losses):.2f} "
                  f"| val_rmse={val_rmse:.2f} | val_mae={val_mae:.2f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "rul_lstm.pt"))
    print(f"\nSaved model to {os.path.join(MODELS_DIR, 'rul_lstm.pt')}")
    print(f"Saved scaler to {os.path.join(MODELS_DIR, 'rul_scaler.pkl')}")

    # ---------------------------------------------------------------
    # Official test-set evaluation
    # ---------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    test_rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))
    test_mae = float(np.mean(np.abs(test_preds - y_test)))
    print(f"\nFinal TEST RMSE: {test_rmse:.2f}")
    print(f"Final TEST MAE:  {test_mae:.2f}")

    # ---------------------------------------------------------------
    # Predicted vs actual plot
    # ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(y_test)[::-1]
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[order], label="Actual RUL", marker="o", markersize=3)
    plt.plot(test_preds[order], label="Predicted RUL", marker="x", markersize=3)
    plt.xlabel("Test engine (sorted by actual RUL, descending)")
    plt.ylabel("RUL (cycles)")
    plt.title(f"AeroVision RUL — Predicted vs Actual (Test RMSE={test_rmse:.1f}, MAE={test_mae:.1f})")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "rul_test_predictions.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved prediction plot to {out_path}")


if __name__ == "__main__":
    main()
