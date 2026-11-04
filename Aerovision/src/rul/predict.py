"""
Inference helpers for the trained RUL LSTM model.

Loads the model + scaler saved by train.py and exposes:
    predict_rul(sensor_sequence) -> predicted RUL in cycles
    health_score(predicted_rul)  -> 0-100 health score
    failure_risk_band(predicted_rul) -> "HIGH" | "MEDIUM" | "LOW"

Risk band naming matches AeroVision's existing severity.py convention
(same band names, same idea: a simple, explainable rubric a judge can
verify at a glance) — kept as a plain function here rather than
importing severity.py, since severity.py's bands are about *defect*
urgency (size/confidence/type based) and RUL's bands are about
*engine life remaining*; the underlying scoring logic isn't shared,
only the LOW/MEDIUM/HIGH vocabulary.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .data_prep import load_scaler
from .model import build_lstm_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "rul_lstm.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "rul_scaler.pkl")

_model = None
_scaler = None


def _load_artifacts(num_features: int):
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained RUL model found at {MODEL_PATH}. Run `python src/rul/train.py` first."
            )
        _model = build_lstm_model(input_shape=(None, num_features))
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
    if _scaler is None:
        _scaler = load_scaler(SCALER_PATH)
    return _model, _scaler


def predict_rul(sensor_sequence: np.ndarray) -> float:
    """
    sensor_sequence: array of shape [window_size, num_features],
    already the correct set/order of sensor columns (post
    drop_constant_sensors) but RAW (unscaled) values — this function
    applies the saved scaler itself, so callers don't need to
    duplicate the normalization step used at training time.

    Returns predicted RUL in cycles (float).
    """
    sensor_sequence = np.asarray(sensor_sequence, dtype=np.float32)
    if sensor_sequence.ndim != 2:
        raise ValueError("sensor_sequence must be 2D: [window_size, num_features]")

    num_features = sensor_sequence.shape[1]
    model, scaler = _load_artifacts(num_features)

    scaled = scaler.transform(sensor_sequence)
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0)  # [1, window_size, num_features]

    with torch.no_grad():
        pred = model(x).item()
    return float(pred)


def health_score(predicted_rul: float, max_expected_rul: float = 125) -> float:
    """0-100 health score; predicted_rul at or above max_expected_rul -> 100."""
    return float(min(100.0, (predicted_rul / max_expected_rul) * 100.0))


def failure_risk_band(predicted_rul: float) -> str:
    """HIGH if < 20 cycles remaining, MEDIUM if 20-50, LOW if > 50."""
    if predicted_rul < 20:
        return "HIGH"
    elif predicted_rul <= 50:
        return "MEDIUM"
    else:
        return "LOW"
