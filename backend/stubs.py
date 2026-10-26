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
    try:
        from predict_gait import get_mobility_report as _real_get_mobility_report
    except ImportError:
        _real_get_mobility_report = None
else:
    _real_get_mobility_report = None



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


def get_mobility_report(user_id="SE01", days=14) -> dict:
    if USE_STUBS or _real_get_mobility_report is None:
        return {
            "user_id": user_id,
            "days_analyzed": days,
            "total_active_minutes": round(25.0 * days, 1),
            "current_mobility_score": 88.0,
            "average_mobility_score": 90.0,
            "average_cadence_hz": 1.70,
            "average_stride_variability_cv": 9.5,
            "average_step_regularity": 0.42,
            "average_lateral_sway_rms": 0.22,
            "trend": "stable",
            "slope_points_per_week": 0.0,
            "caregiver_alert": False,
            "recommendation": "Walking mobility is consistent with established personal baseline (stub).",
            "daily_history": [
                {"day": i, "mobility_score": round(89.0 + (i % 3) * 1.5, 1)}
                for i in range(1, days + 1)
            ],
            "placeholder": True,
        }
    try:
        report = dict(_real_get_mobility_report(user_id=user_id, days=days))
        # Attach daily_history if available in reports
        report_file = REPO / "data" / "reports" / "mobility_decline_report.json"
        if report_file.exists():
            try:
                import json
                data = json.loads(report_file.read_text())
                report["daily_history"] = data.get("daily_history", [])[-days:]
            except Exception:
                pass
        return report
    except Exception as e:
        print(f"[MOBILITY] report error: {e}")
        return {
            "user_id": user_id,
            "days_analyzed": days,
            "total_active_minutes": 0.0,
            "current_mobility_score": 0.0,
            "average_mobility_score": 0.0,
            "average_cadence_hz": 0.0,
            "average_stride_variability_cv": 0.0,
            "average_step_regularity": 0.0,
            "average_lateral_sway_rms": 0.0,
            "trend": "unknown",
            "slope_points_per_week": 0.0,
            "caregiver_alert": False,
            "recommendation": f"Report error: {e}",
            "placeholder": True,
        }