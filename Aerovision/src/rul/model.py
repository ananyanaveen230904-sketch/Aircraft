"""
LSTM model for Remaining Useful Life (RUL) prediction, in PyTorch.

Chosen over TensorFlow/Keras to avoid pulling a second deep-learning
framework into AeroVision: `torch` is already a dependency for the
optional YOLOv8 detection backend, so this reuses it rather than
adding TensorFlow on top.

Architecture (as specified):
    LSTM(64, return_sequences=True) -> Dropout(0.2)
    -> LSTM(32) -> Dropout(0.2)
    -> Dense(16, relu) -> Dense(1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RULLSTM(nn.Module):
    def __init__(self, input_size: int, lstm1_hidden: int = 64, lstm2_hidden: int = 32,
                 dense_hidden: int = 16, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, lstm1_hidden, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm1_hidden, lstm2_hidden, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(lstm2_hidden, dense_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dense_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window_size, num_features]
        out, _ = self.lstm1(x)          # -> [batch, window_size, lstm1_hidden]  (return_sequences=True)
        out = self.dropout1(out)
        out, (h_n, _) = self.lstm2(out)  # -> only the final hidden state is used (Keras LSTM(32) default)
        out = h_n[-1]                    # [batch, lstm2_hidden]
        out = self.dropout2(out)
        out = self.relu(self.dense1(out))
        out = self.dense2(out)           # [batch, 1]
        return out.squeeze(-1)           # [batch]


def build_lstm_model(input_shape: tuple) -> RULLSTM:
    """
    input_shape: (window_size, num_features) — matches the Keras-style
    signature from the spec. window_size isn't needed to construct the
    PyTorch model (LSTMs are sequence-length-agnostic), only
    num_features is.
    """
    _, num_features = input_shape
    return RULLSTM(input_size=num_features)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.functional.mse_loss(pred, target))


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, target)
