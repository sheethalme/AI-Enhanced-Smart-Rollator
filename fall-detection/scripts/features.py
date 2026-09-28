"""Per-window features from pooled_windows.parquet -> data/processed/features[_clip2g].parquet.

--clip2g clips every accelerometer axis to +/-2 g before computing features, i.e.
emulates MobiFall's saturated phone accelerometer on SisFall, so both datasets
see the same sensor range (MobiFall is unaffected -- it is already clipped).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from common import FS, PROCESSED

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
MAGS = ["amag", "gmag"]
META = ["dataset", "subject", "trial", "start", "label", "label3"]
CORR_PAIRS = [("ax", "ay"), ("ax", "az"), ("ay", "az"), ("gx", "gy"), ("gx", "gz"), ("gy", "gz")]
ACC_CLIP = 2.0


RAW_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]   # input order: g, deg/s


def load_signals(pool, clip):
    return derive_signals({c: np.stack(pool[c].to_numpy()) for c in RAW_CHANNELS}, clip)


def derive_signals(raw, clip):
    """raw: {channel: (n_windows, 125)} for RAW_CHANNELS -> adds amag/gmag, optional clip."""
    sig = {c: np.asarray(raw[c], dtype=np.float64) for c in RAW_CHANNELS}
    if clip:
        for c in ["ax", "ay", "az"]:
            sig[c] = np.clip(sig[c], -ACC_CLIP, ACC_CLIP)
    sig["amag"] = np.sqrt(sig["ax"] ** 2 + sig["ay"] ** 2 + sig["az"] ** 2)
    sig["gmag"] = np.sqrt(sig["gx"] ** 2 + sig["gy"] ** 2 + sig["gz"] ** 2)
    return sig


def channel_features(x, name):
    """x: (n_windows, 125)."""
    return {
        f"{name}_mean": x.mean(1), f"{name}_std": x.std(1),
        f"{name}_min": x.min(1), f"{name}_max": x.max(1),
        f"{name}_rms": np.sqrt((x ** 2).mean(1)),
        f"{name}_energy": (x ** 2).sum(1) / x.shape[1],
        f"{name}_peak": np.abs(x).max(1),
        f"{name}_jerk": np.abs(np.diff(x, axis=1)).mean(1) * FS,
        f"{name}_kurt": kurtosis(x, axis=1), f"{name}_skew": skew(x, axis=1),
    }


def row_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.divide((a * b).sum(1), den, out=np.zeros(len(a)), where=den > 0)


def dominant_freq(x):
    spec = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], d=1 / FS)
    return freqs[spec[:, 1:].argmax(1) + 1]          # skip DC


def extract_features(sig):
    """sig from derive_signals -> DataFrame, one row per window, fixed column order."""
    feats = {}
    for c in AXES + MAGS:
        feats.update(channel_features(sig[c], c))
    feats["acc_sma"] = (np.abs(sig["ax"]) + np.abs(sig["ay"]) + np.abs(sig["az"])).mean(1)
    feats["gyro_sma"] = (np.abs(sig["gx"]) + np.abs(sig["gy"]) + np.abs(sig["gz"])).mean(1)
    for a, b in CORR_PAIRS:
        feats[f"corr_{a}_{b}"] = row_corr(sig[a], sig[b])
    feats["amag_domfreq"] = dominant_freq(sig["amag"])
    return pd.DataFrame(feats).fillna(0.0)            # constant windows -> NaN kurt/skew


def build(pool, clip):
    X = extract_features(load_signals(pool, clip))
    return pd.concat([pool[META].reset_index(drop=True), X], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip2g", action="store_true")
    args = ap.parse_args()
    pool = pd.read_parquet(PROCESSED / "pooled_windows.parquet")
    df = build(pool, args.clip2g)
    out = PROCESSED / ("features_clip2g.parquet" if args.clip2g else "features.parquet")
    df.to_parquet(out, index=False)
    print(f"{out.name}: {len(df):,} windows x {df.shape[1] - len(META)} features")


if __name__ == "__main__":
    main()
