"""Stage 0: inspect raw SisFall (Enhanced, 3-class) and MobiFall v2.0. No conversion."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SIS, MOBI = RAW / "sisfall_kaggle_windows", RAW / "mobifall"   # Kaggle tensors, superseded
FALL_CODES = {"FOL", "FKL", "BSC", "SDL"}


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------- SisFall
def load_sisfall(split):
    y = np.fromfile(SIS / f"y_{split}_3", dtype=np.uint8)
    n = len(y) // 3
    x = np.fromfile(SIS / f"x_{split}_3", dtype=np.float32)
    assert x.size % n == 0
    return x.reshape(n, -1, 6), y.reshape(n, 3)


def inspect_sisfall():
    header("SISFALL (Enhanced, 3-class)")
    print("Files:", sorted(p.name for p in SIS.iterdir()))
    print("Format: headerless binary. x_* = float32, y_* = uint8 one-hot (3 cols).")
    print("        No subject / trial / timestamp info anywhere in the files.")
    print("weights_3.txt:", SIS.joinpath("weights_3.txt").read_text().split())

    for split in ["train", "val", "test"]:
        x, y = load_sisfall(split)
        print(f"  {split:5s}: x {x.shape}  y {y.shape}  one-hot valid={bool((y.sum(1) == 1).all())}"
              f"  class counts={y.sum(0).tolist()}")

    x, y = load_sisfall("train")
    cls = y.argmax(1)
    cols = ["c0", "c1", "c2", "c3", "c4", "c5"]
    flat = pd.DataFrame(x.reshape(-1, 6), columns=cols)
    print("\nColumns: none named; 6 channels per sample (presumably ax ay az gx gy gz).")
    print("First 5 samples of window 0:")
    print(flat.head().to_string())
    print("\nPer-channel stats (train):")
    print(flat.agg(["min", "max", "mean", "std"]).T.round(4).to_string())

    amag = np.sqrt((x[:, :, :3] ** 2).sum(-1))
    still = amag.std(1) < 0.002
    g_still = np.median(amag[still].mean(1))
    print(f"\nGravity check: {still.sum()} near-still windows, median amag = {g_still:.4f}"
          f"  (expected 1.0 g) -> implied scale x{1 / g_still:.2f}")
    print("Per-class amag peak (median / p95) and gyro-mag peak median:")
    for c in range(3):
        m = cls == c
        gpk = np.sqrt((x[m][:, :, 3:] ** 2).sum(-1)).max(1)
        print(f"  class {c}: n={m.sum():6d}  amag peak {np.median(amag[m].max(1)):.3f} / "
              f"{np.percentile(amag[m].max(1), 95):.3f}   gyro peak {np.median(gpk):.3f}")

    print("\nSampling rate: no timestamps. 256 samples/window; SisFall native rate is 200 Hz"
          " -> 1.28 s/window IF not resampled.")
    shifts = [min(range(1, 256), key=lambda k: np.abs(x[i, k:] - x[i + 1, :256 - k]).mean())
              for i in range(20)]
    print("Consecutive-window best-overlap shifts (random => shuffled, not trial order):", shifts)
    print("Label sequence, first 80 windows:", "".join(map(str, cls[:80])))
    print("Fall/ADL encoding: one-hot y column per WINDOW (not per sample).")


# ---------------------------------------------------------------- MobiFall
def read_mobifall(path):
    lines = path.read_text().splitlines()
    start = lines.index("@DATA") + 1
    df = pd.read_csv(path, skiprows=start, header=None, names=["t_ns", "x", "y", "z"])
    return df


def inspect_mobifall():
    header("MOBIFALL v2.0")
    files = sorted(p for p in MOBI.rglob("*.txt") if p.name != "Readme.txt")
    pat = re.compile(r"([A-Z]{3})_(acc|gyro|ori)_(\d+)_(\d+)\.txt")
    meta = pd.DataFrame([pat.fullmatch(p.name).groups() + (p.parent.parent.name,) for p in files],
                        columns=["code", "sensor", "subject", "trial", "folder"])
    print("Layout: subN/{ADL|FALLS}/<CODE>/<CODE>_<sensor>_<subject>_<trial>.txt")
    print(f"{len(files)} data files, subjects: {sorted(meta.subject.astype(int).unique())}")
    print("Sensor counts:", dict(Counter(meta.sensor)))
    by = meta[meta.sensor == "acc"].groupby(["folder", "code"]).size()
    print("acc trials per (folder, code):\n" + by.to_string())
    adl_subj = sorted(meta[meta.folder == "ADL"].subject.astype(int).unique())
    print("Subjects with ADL recordings:", adl_subj)
    print("Fall/ADL encoding: folder name (FALLS vs ADL) == 3-letter code in",
          sorted(FALL_CODES), "-> whole-trial label, no per-sample labels.")
    print("Placement: not in files; MobiFall = smartphone in trouser pocket (all trials).")

    # sample rows + stats across ALL acc and gyro files
    for sensor in ["acc", "gyro"]:
        sfiles = [p for p in files if f"_{sensor}_" in p.name]
        ex = read_mobifall(sfiles[0])
        print(f"\n--- {sensor}: header comment ---")
        print("\n".join(l for l in sfiles[0].read_text().splitlines()[:2]))
        print(f"Example {sfiles[0].relative_to(MOBI)} first 5 rows:")
        print(ex.head().to_string())

        frames, dts, rates = [], [], []
        for p in sfiles:
            d = read_mobifall(p)
            frames.append(d[["x", "y", "z"]])
            dt = np.diff(d.t_ns.values) / 1e6  # ms
            dts.append(dt)
            span = (d.t_ns.iloc[-1] - d.t_ns.iloc[0]) / 1e9
            rates.append((len(d) - 1) / span if span > 0 else np.nan)
        allv = pd.concat(frames)
        dt = np.concatenate(dts)
        print(f"Stats over all {len(sfiles)} {sensor} files ({len(allv):,} samples):")
        print(allv.agg(["min", "max", "mean", "std"]).T.round(4).to_string())
        mag = np.sqrt((allv ** 2).sum(1))
        print(f"|vector| median = {mag.median():.3f}")
        print(f"dt (ms): median {np.median(dt):.2f}  p5 {np.percentile(dt, 5):.2f}  "
              f"p95 {np.percentile(dt, 95):.2f}  max {dt.max():.1f}  "
              f"non-positive {int((dt <= 0).sum())}")
        print(f"Per-file mean rate (Hz): median {np.nanmedian(rates):.1f}  "
              f"min {np.nanmin(rates):.1f}  max {np.nanmax(rates):.1f}")

    # acc vs gyro alignment in one trial
    a = read_mobifall(MOBI / "sub1/FALLS/FOL/FOL_acc_1_1.txt")
    g = read_mobifall(MOBI / "sub1/FALLS/FOL/FOL_gyro_1_1.txt")
    print(f"\nacc/gyro alignment (sub1 FOL 1): acc {len(a)} rows, gyro {len(g)} rows; "
          f"start offset {(a.t_ns[0] - g.t_ns[0]) / 1e6:.1f} ms, "
          f"end offset {(a.t_ns.iloc[-1] - g.t_ns.iloc[-1]) / 1e6:.1f} ms -> separate clocks per sensor")

    # gravity check on the standing trials
    std = [p for p in files if p.name.startswith("STD_acc")]
    m = [np.sqrt((read_mobifall(p)[["x", "y", "z"]] ** 2).sum(1)).median() for p in std]
    print(f"STD (standing) acc |a| median per trial: {np.round(m, 3).tolist()} (m/s^2?)")
    falls = [p for p in files if p.name[:3] in FALL_CODES and "_acc_" in p.name]
    pk = [np.sqrt((read_mobifall(p)[["x", "y", "z"]] ** 2).sum(1)).max() / 9.81 for p in falls]
    print(f"Fall trials acc peak (after /9.81): median {np.median(pk):.2f} g, "
          f"p5 {np.percentile(pk, 5):.2f}, p95 {np.percentile(pk, 95):.2f}")


if __name__ == "__main__":
    inspect_sisfall()
    inspect_mobifall()
