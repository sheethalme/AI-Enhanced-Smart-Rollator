"""Shared constants + helpers for the harmonize / window / pool scripts."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FS = 50                       # target rate, Hz
DT = 1.0 / FS
WIN = 125                     # 2.5 s
STRIDE = 62                   # ~50 % overlap
MAX_GAP_S = 0.1               # don't interpolate across raw gaps longer than this
G = 9.81
RAD2DEG = 57.2958

# label3 codes (shared by both datasets)
NONFALL, PREFALL, FALL = 0, 1, 2

SAMPLE_COLS = ["t", "ax", "ay", "az", "gx", "gy", "gz", "amag",
               "label", "label3", "dataset", "subject", "placement", "trial"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "amag"]


def resample_segments(streams, t_start, t_end):
    """Resample several (t_seconds, values[n, k]) streams onto one uniform 50 Hz grid.

    Interpolates on the real timestamps. Grid points that fall inside a raw gap
    longer than MAX_GAP_S in ANY stream are dropped, and the grid is split there.
    Returns a list of (grid_t, stacked_values) contiguous segments.
    """
    grid = np.arange(t_start, t_end, DT)
    ok = np.ones(len(grid), bool)
    cols = []
    for t, v in streams:
        # position of each grid point between raw samples -> gap length around it
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        ok &= (t[idx] - t[idx - 1]) <= MAX_GAP_S
        cols.append(np.column_stack([np.interp(grid, t, v[:, j]) for j in range(v.shape[1])]))
    vals = np.hstack(cols)

    # split into runs of consecutive ok points
    segs, edges = [], np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        segs.append((grid[a:b], vals[a:b]))
    return segs


def to_frame(t, vals6, label3, meta):
    """Build a sample-level DataFrame in the shared schema."""
    df = pd.DataFrame(vals6, columns=["ax", "ay", "az", "gx", "gy", "gz"])
    df.insert(0, "t", np.round(t - t[0], 4))
    df["amag"] = np.sqrt(df.ax ** 2 + df.ay ** 2 + df.az ** 2)
    df["label3"] = np.asarray(label3, dtype=np.int8)
    df["label"] = (df.label3 == FALL).astype(np.int8)   # prefall -> 0 for now
    for k, v in meta.items():
        df[k] = v
    return df[SAMPLE_COLS]
