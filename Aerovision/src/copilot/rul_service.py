"""
Bridges the existing LSTM RUL module (src/rul) to the copilot.

Takes a NASA C-MAPSS FD001 *test engine id*, builds its last 30-cycle sensor
window exactly like src/rul/train.py does, and calls src.rul.predict.
Returns {"error": ...} (never raises) when torch or the trained weights
(models/rul_lstm.pt, created by `python src/rul/train.py`) are missing.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from . import config

WINDOW = 30
_cache: Dict[str, Any] = {}


def _load():
    if "test" in _cache:
        return _cache
    from src.rul.data_prep import SENSOR_COLUMNS, drop_constant_sensors, load_cmapss
    train = load_cmapss(os.path.join(config.CMAPSS_DIR, "train_FD001.txt"))
    test = load_cmapss(os.path.join(config.CMAPSS_DIR, "test_FD001.txt"))
    _, dropped = drop_constant_sensors(train)
    _cache["features"] = [c for c in SENSOR_COLUMNS if c not in dropped]
    _cache["test"] = test
    return _cache


def list_engine_ids() -> list:
    try:
        return sorted(int(i) for i in _load()["test"]["engine_id"].unique())
    except Exception:
        return []


def predict_engine(engine_id: int) -> Dict[str, Any]:
    try:
        c = _load()
        from src.rul.predict import failure_risk_band, health_score, predict_rul
        df = c["test"]
        eng = df[df["engine_id"] == int(engine_id)].sort_values("cycle")
        if eng.empty:
            return {"error": f"engine_id {engine_id} not found (valid: 1-{int(df['engine_id'].max())})"}
        feats = eng[c["features"]].to_numpy()
        if len(feats) < WINDOW:
            import numpy as np
            feats = np.vstack([np.repeat(feats[:1], WINDOW - len(feats), axis=0), feats])
        rul = predict_rul(feats[-WINDOW:])
        return {
            "engine_id": int(engine_id),
            "rul_cycles": round(float(rul), 1),
            "health_score": round(health_score(rul), 1),
            "failure_risk": failure_risk_band(rul),
            "source": "LSTM on NASA C-MAPSS FD001 (last 30 cycles)",
        }
    except FileNotFoundError as e:
        return {"error": f"{e} Train the RUL model first: python src/rul/train.py"}
    except ImportError as e:
        return {"error": f"Missing dependency for RUL model: {e}. pip install torch"}
    except Exception as e:  # keep the agent alive
        return {"error": f"RUL prediction failed: {e}"}
