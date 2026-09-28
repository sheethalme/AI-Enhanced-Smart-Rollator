"""MobiFall v2.0 -> data/processed/mobifall.parquet (sample-level, 50 Hz, g + deg/s).

Raw: per-sensor text files, t in ns, acc m/s^2 (~85 Hz, irregular), gyro rad/s (~200 Hz).
Acc and gyro have separate clocks -> each is interpolated onto a shared 50 Hz grid
covering only the time both sensors were recording. Placement = trouser pocket.

Labels: MobiFall only labels whole trials. A 10 s fall trial is mostly standing
before and lying after, so per-sample FALL is assigned to a window around the
impact (peak amag); everything else in the trial is NONFALL. See FALL_PRE/FALL_POST.
"""
import re

import numpy as np
import pandas as pd

from common import FALL, G, NONFALL, PROCESSED, RAD2DEG, RAW, resample_segments, to_frame

SRC = RAW / "mobifall"
FALL_CODES = {"FOL", "FKL", "BSC", "SDL"}
TRIM_START_S = 0.1          # first gyro samples of each trial contain junk spikes
SLOW_DT = 0.15              # acc often starts in a ~5 Hz mode (dt 0.2 s) before going fast
FALL_PRE, FALL_POST = 0.2, 0.2   # s around impact; matches SisFall-enhanced median FALL span
NAME = re.compile(r"([A-Z]{3})_acc_(\d+)_(\d+)\.txt")


def read(path):
    lines = path.read_text().splitlines()
    df = pd.read_csv(path, skiprows=lines.index("@DATA") + 1, header=None,
                     names=["t", "x", "y", "z"])
    df["t"] = df.t / 1e9
    df = df.drop_duplicates("t").sort_values("t")   # a few non-increasing stamps
    return df.t.to_numpy(), df[["x", "y", "z"]].to_numpy(float)


def fast_start(t):
    """Time at which the acc stream leaves its initial slow (~5 Hz) mode."""
    head = np.flatnonzero(np.diff(t[: len(t) // 3]) > SLOW_DT)
    return t[head[-1] + 1] if len(head) else t[0]


def fall_label3(t, amag):
    """FALL around the single biggest impact in the trial, NONFALL elsewhere."""
    tp = t[np.argmax(amag)]
    return np.where((t >= tp - FALL_PRE) & (t <= tp + FALL_POST), FALL, NONFALL)


def harmonize_trial(acc_path):
    code, subj, trial = NAME.fullmatch(acc_path.name).groups()
    gyro_path = acc_path.with_name(acc_path.name.replace("_acc_", "_gyro_"))
    ta, a = read(acc_path)
    tg, g = read(gyro_path)
    a = a / G                       # m/s^2 -> g
    g = g * RAD2DEG                 # rad/s -> deg/s

    t0 = max(fast_start(ta), tg[0]) + TRIM_START_S
    t1 = min(ta[-1], tg[-1])
    is_fall = code in FALL_CODES
    frames = []
    for i, (t, v) in enumerate(resample_segments([(ta, a), (tg, g)], t0, t1)):
        amag = np.sqrt((v[:, :3] ** 2).sum(1))
        lab3 = fall_label3(t, amag) if is_fall else np.full(len(t), NONFALL)
        frames.append(to_frame(t, v, lab3, {
            "dataset": "mobifall", "subject": f"mobi_{int(subj):02d}",
            "placement": "pocket", "trial": f"{code}_{subj}_{trial}_s{i}"}))
    return frames


def main():
    acc_files = sorted(p for p in SRC.rglob("*_acc_*.txt"))
    frames = [f for p in acc_files for f in harmonize_trial(p)]
    df = pd.concat(frames, ignore_index=True)
    for c in ["dataset", "subject", "placement", "trial"]:
        df[c] = df[c].astype("category")
    out = PROCESSED / "mobifall.parquet"
    df.to_parquet(out, index=False)

    trials = df.groupby("trial", observed=True)
    print(f"MobiFall: {len(acc_files)} raw trials -> {trials.ngroups} segments, "
          f"{len(df):,} samples @ 50 Hz -> {out.name}")
    print(f"  subjects {df.subject.nunique()}, label3 counts {df.label3.value_counts().sort_index().to_dict()}")
    print(f"  amag median (all) {df.amag.median():.3f} g, max {df.amag.max():.2f} g")


if __name__ == "__main__":
    main()
