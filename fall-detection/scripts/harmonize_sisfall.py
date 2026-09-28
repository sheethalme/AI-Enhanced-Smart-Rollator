"""SisFall + Musci et al. temporal annotation -> data/processed/sisfall.parquet.

Raw: data/raw/sisfall/SisFall_dataset/<SUBJ>/<CODE>_<SUBJ>_<Rnn>.txt, 9 int columns
     (ADXL345 xyz, ITG3200 xyz, MMA8451Q xyz), 200 Hz, raw ADC counts, waist-worn.
Labels: data/raw/sisfall/SisFall_enhanced/<SUBJ>/<same name>.txt, one value per raw
     row: 0 BKG, 1 ALERT (prefall), 2 FALL.
We use ADXL345 (+/-16 g, 13 bit) and ITG3200 (+/-2000 deg/s, 16 bit).
"""
import re

import numpy as np
import pandas as pd

from common import FALL, PROCESSED, RAW, resample_segments, to_frame

SRC = RAW / "sisfall" / "SisFall_dataset"
LAB = RAW / "sisfall" / "SisFall_enhanced"
FS_RAW = 200.0
ACC_SCALE = (2 * 16) / 2 ** 13      # ADC -> g
GYRO_SCALE = (2 * 2000) / 2 ** 16   # ADC -> deg/s
MAX_ROW_DIFF = 5
NAME = re.compile(r"([DF]\d\d)_(S[AE]\d\d)_(R\d\d)\.txt")


def read_raw(path):
    rows = [l.strip().rstrip(";") for l in path.read_text().splitlines()]
    rows = [r for r in rows if r]
    return np.array([[int(v) for v in r.split(",")] for r in rows])


def resample_labels(t_raw, lab, t_new):
    """Label of each 50 Hz sample = max label in its +/-10 ms neighbourhood, so short
    ALERT/FALL spans are never dropped by the 4x down-sampling."""
    lo = np.searchsorted(t_raw, t_new - 0.01)
    hi = np.searchsorted(t_raw, t_new + 0.01, side="right")
    return np.array([lab[a:b].max() for a, b in zip(lo, hi)])


def harmonize_trial(path, lab_path):
    code, subj, trial = NAME.fullmatch(path.name).groups()
    raw = read_raw(path)
    lab = np.loadtxt(lab_path, dtype=np.int8)
    if abs(len(lab) - len(raw)) > MAX_ROW_DIFF:
        raise ValueError(f"{path.name}: {len(raw)} raw rows vs {len(lab)} labels")
    n = min(len(raw), len(lab))              # a few files are off by 1 row
    raw, lab = raw[:n], lab[:n]
    vals = np.hstack([raw[:, 0:3] * ACC_SCALE, raw[:, 3:6] * GYRO_SCALE])
    t = np.arange(len(raw)) / FS_RAW          # no timestamps in SisFall: nominal 200 Hz
    (tg, v), = resample_segments([(t, vals)], t[0], t[-1] + 1e-9)
    lab3 = resample_labels(t, lab, tg)
    return to_frame(tg, v, lab3, {"dataset": "sisfall", "subject": subj,
                                  "placement": "waist", "trial": f"{code}_{subj}_{trial}"})


def main():
    frames, missing, bad = [], [], []
    for p in sorted(SRC.rglob("*.txt")):
        if not NAME.fullmatch(p.name):
            continue                               # Readme etc.
        lp = LAB / p.parent.name / p.name
        if not lp.exists():
            missing.append(p.name)
            continue
        try:
            frames.append(harmonize_trial(p, lp))
        except ValueError as e:
            bad.append(str(e))
    df = pd.concat(frames, ignore_index=True)
    for c in ["dataset", "subject", "placement", "trial"]:
        df[c] = df[c].astype("category")
    out = PROCESSED / "sisfall.parquet"
    df.to_parquet(out, index=False)

    print(f"SisFall: {len(frames)} trials, {len(df):,} samples @ 50 Hz -> {out.name}")
    print(f"  skipped (no annotation file): {missing}")
    print(f"  skipped (raw/label length mismatch > {MAX_ROW_DIFF} rows): {bad}")
    print(f"  subjects {df.subject.nunique()}, label3 counts "
          f"{df.label3.value_counts().sort_index().to_dict()}")
    print(f"  amag median (all) {df.amag.median():.3f} g, max {df.amag.max():.2f} g")


if __name__ == "__main__":
    main()
