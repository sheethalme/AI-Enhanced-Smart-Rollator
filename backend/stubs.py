"""Model adapters behind the team's contract signatures.

predict_fall(window[125,6])              -> {"fall_prob": float}                     (Member 1)
predict_gait(window[125,6], user_id=None) -> {"instability_score": float, ...extras}  (Member 2)

Set the environment variable USE_STUBS=1 to run with fake models (useful if a model fails to load).
"""
import os
import sys
from pathlib import Path

import numpy as np

USE_STUBS = os.environ.get("USE_STUBS") == "1"

if not USE_STUBS:
    # Works whether backend/ sits inside the repo (final layout) or next to it (current layout).
    _ABOVE_BACKEND = Path(__file__).resolve().parent.parent
    REPO = (_ABOVE_BACKEND if (_ABOVE_BACKEND / "predict_gait.py").exists()
            else _ABOVE_BACKEND / "AI-Enhanced-Smart-Rollator")
    for _p in (REPO, REPO / "fall-detection" / "scripts"):
        if str(_p) not in sys.path:
            sys.path.append(str(_p))
    from predict import predict_fall as _real_predict_fall          # Member 1
    from predict_gait import predict_gait as _real_predict_gait    # Member 2


def _mag(window):
    w = np.asarray(window)
    return np.sqrt((w[:, :3] ** 2).sum(axis=1))


def predict_fall(window) -> dict:
    if USE_STUBS:
        peak = float(_mag(window).max())
        return {"fall_prob": 0.95 if peak > 2.5 else min(0.5, peak / 6)}
    try:
        return {"fall_prob": float(_real_predict_fall(window)["probability"])}
    except ValueError as e:                      # wrong shape / NaN: skip window, keep the stream alive
        print(f"[FALL] window rejected: {e}")
        return {"fall_prob": 0.0}


def predict_gait(window, user_id=None) -> dict:
    if USE_STUBS:
        return {"instability_score": float(min(1.0, _mag(window).std() / 0.5))}
    try:
        # Returns Member 2's full dict (instability_score, cadence, mobility_score, ...).
        # main.py only reads ["instability_score"]; the extra keys are there for the dashboard later.
        return _real_predict_gait(window, user_id)
    except ValueError as e:
        print(f"[GAIT] window rejected: {e}")
        return {"instability_score": 0.0}