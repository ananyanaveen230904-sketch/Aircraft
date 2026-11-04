"""
Data loading and preprocessing for the NASA C-MAPSS FD001 turbofan
degradation dataset, used to train the LSTM Remaining Useful Life
(RUL) model.

File format (verified against the actual downloaded FD001 files):
    26 whitespace-separated columns, no header row, with a trailing
    space at the end of each line (which produces a spurious 27th
    empty column if not stripped):

        engine_id, cycle, op_setting_1, op_setting_2, op_setting_3,
        sensor_1 ... sensor_21
"""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss(path: str) -> pd.DataFrame:
    """
    Read a C-MAPSS train_*.txt or test_*.txt file into a DataFrame
    with named columns. Handles the trailing-whitespace-per-line quirk
    of the raw files (which otherwise yields an extra all-NaN column).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMN_NAMES)]  # drop any spurious trailing column
    df.columns = COLUMN_NAMES
    return df


def load_rul_labels(path: str) -> np.ndarray:
    """Read RUL_FD001.txt — one true RUL value per test engine, in engine order."""
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()


def drop_constant_sensors(df: pd.DataFrame, threshold: float = 1e-5) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop sensor columns whose variance across the whole dataset is
    below `threshold` — these carry no degradation signal (they're
    constant for this operating-condition subset, e.g. commonly
    sensors 1, 5, 6, 10, 16, 18, 19 in FD001) and would otherwise just
    add noise/dimensionality to the LSTM input.

    Verified against variance, not hardcoded, so this stays correct
    if run against a different C-MAPSS subset (FD002-4) later.
    """
    variances = df[SENSOR_COLUMNS].var()
    dropped = variances[variances < threshold].index.tolist()
    kept_df = df.drop(columns=dropped)
    print(f"[drop_constant_sensors] Dropped {len(dropped)} near-constant sensor(s): {dropped}")
    return kept_df, dropped


def normalize_sensors(
    df: pd.DataFrame,
    sensor_columns: List[str],
    scaler: MinMaxScaler | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Min-max normalize the given sensor columns.

    fit=True (training data): fits a new scaler on this data.
    fit=False (test data): reuses the scaler passed in — must be
        the one fitted on training data, never re-fit on test data,
        or train/test distributions leak into each other.
    """
    out = df.copy()
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(out[sensor_columns].to_numpy())
        out[sensor_columns] = scaler.transform(out[sensor_columns].to_numpy())
    else:
        if scaler is None:
            raise ValueError("normalize_sensors: scaler must be provided when fit=False")
        out[sensor_columns] = scaler.transform(out[sensor_columns].to_numpy())
    return out, scaler


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> MinMaxScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_rul_labels(df: pd.DataFrame, ceiling: int = 125) -> pd.DataFrame:
    """
    Training-data RUL labels: for each row, RUL = (max cycle observed
    for that engine) - (current cycle), capped at `ceiling`.

    The cap reflects standard C-MAPSS practice: an engine far from
    failure has an ambiguous/roughly-constant true degradation rate,
    so RUL targets are clipped rather than left to grow unbounded,
    which stabilizes training.
    """
    out = df.copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=ceiling)
    return out


def create_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    window_size: int = 30,
    label_column: str = "RUL",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-cycle rows into fixed-length sliding-window sequences
    per engine.

    Returns X of shape [num_sequences, window_size, num_features] and
    y of shape [num_sequences] (the RUL at the END of each window).

    Engines with fewer cycles than window_size are SKIPPED (not
    padded). Rationale: padding a short/noisy sequence (with zeros or
    repeated first-rows) distorts the gradient signal the LSTM learns
    from, for a dataset where padding would rarely help anyway —
    verified against this data: the shortest train engine has 128
    cycles and the shortest test engine has 31 cycles, both comfortably
    above window_size=30, so skipping loses effectively no data here.
    """
    X, y = [], []
    for _, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        n = len(engine_df)
        if n < window_size:
            continue
        feats = engine_df[feature_columns].to_numpy()
        labels = engine_df[label_column].to_numpy() if label_column in engine_df else None
        for start in range(0, n - window_size + 1):
            end = start + window_size
            X.append(feats[start:end])
            if labels is not None:
                y.append(labels[end - 1])  # RUL at the end of the window
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32) if y else np.array([], dtype=np.float32)
    return X, y


def create_test_sequences_last_window(
    df: pd.DataFrame,
    feature_columns: List[str],
    window_size: int = 30,
) -> Tuple[np.ndarray, List[int]]:
    """
    For test-set evaluation: C-MAPSS test engines are truncated
    mid-life and each has exactly ONE true RUL value (from
    RUL_FD001.txt), corresponding to the LAST cycle recorded for that
    engine. So unlike training (many sliding windows per engine), test
    evaluation uses just the final `window_size` cycles per engine.

    Returns X of shape [num_engines, window_size, num_features] and
    the list of engine_ids in the same order (so predictions can be
    matched back to RUL_FD001.txt, which is ordered by engine_id).
    """
    X = []
    engine_ids = []
    for engine_id, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        feats = engine_df[feature_columns].to_numpy()
        if len(feats) < window_size:
            # Pad by repeating the earliest row — only needed here (not
            # in training) because every test engine MUST produce
            # exactly one prediction to match RUL_FD001.txt; there's no
            # option to skip.
            pad = np.repeat(feats[:1], window_size - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-window_size:])
        engine_ids.append(engine_id)
    return np.array(X, dtype=np.float32), engine_ids
