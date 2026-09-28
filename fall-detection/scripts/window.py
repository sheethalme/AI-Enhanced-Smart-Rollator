"""Cut 2.5 s / 50 % overlap windows per trial from a sample-level parquet.

One row per window. Signals are kept as list columns (125 floats per channel) so
features.py can compute anything later without going back to the samples.

Window labelling (per-sample label3 from harmonize_*):
  - FALL      if the window holds >= MIN_FALL_SAMPLES FALL samples
                (or the whole FALL span of the trial, if that span is shorter)
  - dropped   if it holds only a sliver of FALL (impact cut by the window edge) --
                ambiguous, neither a clean fall nor a clean non-fall
  - PREFALL   else if it holds >= MIN_ALERT_SAMPLES ALERT samples (label3 only)
  - NONFALL   otherwise
"""
import numpy as np
import pandas as pd

from common import CHANNELS, FALL, NONFALL, PREFALL, STRIDE, WIN

MIN_FALL_SAMPLES = 10    # 0.2 s ~ half of a typical SisFall-enhanced FALL span
MIN_ALERT_SAMPLES = 10
META = ["dataset", "subject", "placement", "trial"]


def window_trial(g):
    lab3 = g.label3.to_numpy()
    n_fall_trial = int((lab3 == FALL).sum())
    need_fall = min(MIN_FALL_SAMPLES, n_fall_trial) if n_fall_trial else 1
    sig = {c: g[c].to_numpy(np.float32) for c in CHANNELS}
    meta = {k: g[k].iloc[0] for k in META}
    rows, dropped = [], 0
    for s in range(0, len(g) - WIN + 1, STRIDE):
        w = lab3[s:s + WIN]
        n_fall, n_alert = int((w == FALL).sum()), int((w == PREFALL).sum())
        if n_fall >= need_fall:
            l3 = FALL
        elif n_fall > 0:
            dropped += 1
            continue
        elif n_alert >= MIN_ALERT_SAMPLES:
            l3 = PREFALL
        else:
            l3 = NONFALL
        rows.append({**meta, "start": s, "label": int(l3 == FALL), "label3": l3,
                     **{c: sig[c][s:s + WIN] for c in CHANNELS}})
    return rows, dropped


def window_dataset(df):
    rows, dropped, short = [], 0, 0
    for _, g in df.groupby("trial", observed=True, sort=False):
        if len(g) < WIN:
            short += 1
            continue
        r, d = window_trial(g)
        rows += r
        dropped += d
    out = pd.DataFrame(rows)
    return out, {"windows": len(out), "dropped_partial_fall": dropped, "trials_too_short": short}
