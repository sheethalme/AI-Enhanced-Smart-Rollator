"""Fit the production fall model on 100 % of the pooled windows (no holdout).

Same RandomForest (train.model) and features (features.py) as the evaluations.
Chosen variant: accel clipped to +/-2 g + all 89 features -- the most balanced
cross-dataset result (train.py --clip2g: 95.7 % / 84.2 % sensitivity).

-> models/fall/fall_rf.pkl    {"model", "feature_names", "config"} (joblib)
-> models/fall/config.json    same config + feature names, human-readable
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from common import FS, PROCESSED, ROOT, STRIDE, WIN
from features import ACC_CLIP, RAW_CHANNELS
from train import feature_sets, model

OUT = ROOT / "models" / "fall"
CLIP2G = True
FEATURE_SET = "all"
THRESHOLD = 0.5


def main():
    df = pd.read_parquet(PROCESSED / ("features_clip2g.parquet" if CLIP2G else "features.parquet"))
    cols = feature_sets(df)[FEATURE_SET]
    rf = model().fit(df[cols], df.label)

    config = {
        "sample_rate_hz": FS, "window_samples": WIN, "window_seconds": WIN / FS,
        "train_stride_samples": STRIDE, "input_channels": RAW_CHANNELS,
        "input_units": {"ax/ay/az": "g", "gx/gy/gz": "deg/s"},
        "acc_clip_g": ACC_CLIP if CLIP2G else None, "feature_set": FEATURE_SET,
        "threshold": THRESHOLD, "positive_label": "fall (label=1)",
        "trained_on": {"windows": int(len(df)), "falls": int(df.label.sum()),
                       "subjects": int(df.subject.nunique()),
                       "datasets": sorted(df.dataset.astype(str).unique())},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "feature_names": cols, "config": config}, OUT / "fall_rf.pkl")
    (OUT / "config.json").write_text(json.dumps({**config, "feature_names": cols}, indent=2))
    print(f"Trained on {len(df):,} windows ({df.label.sum():,} falls, {df.subject.nunique()} subjects), "
          f"{len(cols)} features -> {OUT.relative_to(ROOT)}/fall_rf.pkl, config.json")


if __name__ == "__main__":
    main()
