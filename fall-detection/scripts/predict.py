"""Fall inference on a single IMU window, using models/fall/fall_rf.pkl.

    from predict import predict_fall
    predict_fall(window)  # -> {"fall": bool, "probability": float}
"""
import warnings
from pathlib import Path

import joblib
import numpy as np

from features import RAW_CHANNELS, derive_signals, extract_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fall" / "fall_rf.pkl"

_bundle = joblib.load(MODEL_PATH)          # loaded once, at import
_model = _bundle["model"]
_model.n_jobs = 1                          # single-window calls: skip thread start-up
FEATURE_NAMES = _bundle["feature_names"]
CONFIG = _bundle["config"]
WINDOW_SAMPLES = CONFIG["window_samples"]


def predict_fall(window):
    """Classify one IMU window as fall / non-fall.

    Parameters
    ----------
    window : array-like, shape (125, 6)
        2.5 s of IMU data sampled uniformly at 50 Hz. Columns, in this order:
        [ax, ay, az, gx, gy, gz]
          ax, ay, az : acceleration INCLUDING gravity, in g  (1 g = 9.81 m/s^2;
                       a device at rest reads |a| ~= 1.0)
          gx, gy, gz : angular velocity in deg/s  (rad/s * 57.2958)
        Do not pass amag -- it is computed here. Acceleration is clipped to
        +/-2 g per axis internally, exactly as in training.

    Returns
    -------
    dict
        {"fall": bool, "probability": float}  probability of fall in [0, 1];
        fall = probability >= CONFIG["threshold"] (0.5).
    """
    w = np.asarray(window, dtype=np.float64)
    if w.shape != (WINDOW_SAMPLES, len(RAW_CHANNELS)):
        raise ValueError(f"window must have shape ({WINDOW_SAMPLES}, {len(RAW_CHANNELS)}) "
                         f"[{', '.join(RAW_CHANNELS)}], got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError("window contains NaN or inf")

    sig = derive_signals({c: w[None, :, i] for i, c in enumerate(RAW_CHANNELS)},
                         clip=CONFIG["acc_clip_g"] is not None)
    with warnings.catch_warnings():     # flat channels -> scipy kurt/skew warning; NaN -> 0 as in training
        warnings.simplefilter("ignore", RuntimeWarning)
        X = extract_features(sig)[FEATURE_NAMES]
    p = float(_model.predict_proba(X)[0, list(_model.classes_).index(1)])
    return {"fall": p >= CONFIG["threshold"], "probability": p}


if __name__ == "__main__":
    import pandas as pd

    from common import PROCESSED

    pool = pd.read_parquet(PROCESSED / "pooled_windows.parquet")
    ref = pd.read_parquet(PROCESSED / "features_clip2g.parquet")    # training features, same row order
    picks = pd.concat([pool[(pool.dataset == ds) & (pool.label == lab)].sample(3, random_state=1)
                       for ds in ["sisfall", "mobifall"] for lab in [1, 0]])

    print("Smoke test -- NOTE: these windows were part of the training data, so this checks the\n"
          "plumbing (shapes, units, features, model load), not accuracy.\n")
    print(f"{'dataset':9s} {'trial':16s} {'true':>4s}  {'prob':>6s}  pred   feat-match")
    ok = 0
    for i, r in picks.iterrows():
        window = np.column_stack([r[c] for c in RAW_CHANNELS])           # (125, 6)
        out = predict_fall(window)
        # the features predict_fall builds must equal the ones the model was trained on
        mine = extract_features(derive_signals({c: window[None, :, j] for j, c in enumerate(RAW_CHANNELS)},
                                               clip=True))[FEATURE_NAMES].to_numpy()
        match = np.allclose(mine, ref.loc[i, FEATURE_NAMES].to_numpy(float), atol=1e-9)
        ok += out["fall"] == bool(r.label)
        print(f"{r.dataset:9s} {r.trial:16s} {r.label:4d}  {out['probability']:6.3f}  "
              f"{'FALL' if out['fall'] else 'ok  '}   {match}")
    print(f"\n{ok}/{len(picks)} correct")
