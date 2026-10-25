"""Harmonize SisFall and MobiFall gait and balance trials into standard 50 Hz windows.

Extracts walking (D01, D02, WAL, JOG) and stumble/perturbation (D18, D11, D12, D15, D17) trials.
Saves windowed dataset (125 samples @ 50 Hz, 50% overlap of 62 samples) to gait_windows.parquet.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

from common_gait import (
    FS, WIN, STRIDE, SISFALL_ACC_SCALE, SISFALL_GYRO_SCALE,
    STABLE_WALK, UNSTABLE_WALK, LABEL_NAMES,
    RAW_SISFALL, RAW_MOBIFALL, PROCESSED,
    SISFALL_ACTIVITIES, MOBIFALL_ACTIVITIES,
    G, RAD2DEG, resample_uniform
)

# Regex patterns for file naming
SISFALL_PATTERN = re.compile(r"([DF]\d\d)_(S[AE]\d\d)_(R\d\d)\.txt")
MOBIFALL_PATTERN = re.compile(r"([A-Z]{3})_acc_(\d+)_(\d+)\.txt")


def process_sisfall_trial(raw_path, lab_dir):
    """Load and process a single SisFall trial into 50 Hz windows."""
    match = SISFALL_PATTERN.fullmatch(raw_path.name)
    if not match:
        return []
    code, subj, trial = match.groups()
    cohort = "elderly" if subj.startswith("SE") else "young"

    # Read 200 Hz raw text
    try:
        df_raw = pd.read_csv(raw_path, header=None, comment=";")
    except Exception:
        return []
    if df_raw.shape[1] < 6 or len(df_raw) < 200:
        return []

    # Columns 0..2: ADXL345 acc (g), Columns 3..5: ITG3200 gyro (deg/s)
    raw_acc = df_raw.iloc[:, 0:3].to_numpy(dtype=np.float64) * SISFALL_ACC_SCALE
    raw_gyro = df_raw.iloc[:, 3:6].to_numpy(dtype=np.float64) * SISFALL_GYRO_SCALE

    # Load Musci et al. temporal labels if available
    lab_file = lab_dir / subj / raw_path.name
    n_raw = len(df_raw)
    if lab_file.exists():
        try:
            lab_raw = np.loadtxt(lab_file, dtype=np.int8)
            if len(lab_raw) != n_raw:
                lab_raw = np.resize(lab_raw, n_raw)
        except Exception:
            lab_raw = np.zeros(n_raw, dtype=np.int8)
    else:
        # Default labels: D18 and D11 are unstable perturbation trials
        if code in ["D18", "D11"]:
            lab_raw = np.ones(n_raw, dtype=np.int8)
        else:
            lab_raw = np.zeros(n_raw, dtype=np.int8)

    # Downsample from 200 Hz to 50 Hz (exact factor of 4)
    # Downsampled labels take the maximum in each 4-sample window
    v_acc = raw_acc[::4]
    v_gyro = raw_gyro[::4]
    n_50 = len(v_acc)

    # Max-pooled label across 4 samples
    trim_len = (n_raw // 4) * 4
    lab_50 = lab_raw[:trim_len].reshape(-1, 4).max(axis=1)
    if len(lab_50) < n_50:
        lab_50 = np.pad(lab_50, (0, n_50 - len(lab_50)), mode="edge")
    elif len(lab_50) > n_50:
        lab_50 = lab_50[:n_50]

    # Generate sliding windows
    windows = []
    for start in range(0, n_50 - WIN + 1, STRIDE):
        w_acc = v_acc[start:start + WIN]
        w_gyro = v_gyro[start:start + WIN]
        w_lab = lab_50[start:start + WIN]

        # Determine window label:
        # For D01, D02, D12, D15, D17: standard/controlled walking and transfers -> STABLE
        # For D18, D11: if window overlaps with stumble/alert event (>= 5 samples) -> UNSTABLE
        if code in ["D18", "D11"]:
            label = UNSTABLE_WALK if (w_lab > 0).sum() >= 5 else STABLE_WALK
        else:
            label = STABLE_WALK

        amag = np.sqrt(np.sum(w_acc**2, axis=1))

        windows.append({
            "dataset": "sisfall",
            "subject": subj,
            "cohort": cohort,
            "activity": code,
            "trial": f"{code}_{subj}_{trial}",
            "start": start,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "ax": w_acc[:, 0].tolist(),
            "ay": w_acc[:, 1].tolist(),
            "az": w_acc[:, 2].tolist(),
            "gx": w_gyro[:, 0].tolist(),
            "gy": w_gyro[:, 1].tolist(),
            "gz": w_gyro[:, 2].tolist(),
            "amag": amag.tolist(),
        })
    return windows


def process_mobifall_trial(acc_path):
    """Load and process a single MobiFall walking/jogging trial into 50 Hz windows."""
    match = MOBIFALL_PATTERN.fullmatch(acc_path.name)
    if not match:
        return []
    code, subj, trial = match.groups()
    gyro_path = acc_path.with_name(acc_path.name.replace("_acc_", "_gyro_"))
    if not gyro_path.exists():
        return []

    # Read accelerometer data
    lines_acc = acc_path.read_text().splitlines()
    skip_acc = lines_acc.index("@DATA") + 1
    df_acc = pd.read_csv(acc_path, skiprows=skip_acc, header=None, names=["t", "x", "y", "z"])
    df_acc["t"] = df_acc["t"] / 1e9
    df_acc = df_acc.drop_duplicates("t").sort_values("t")
    ta = df_acc["t"].to_numpy()
    va = df_acc[["x", "y", "z"]].to_numpy(dtype=np.float64) / G  # m/s^2 -> g

    # Read gyroscope data
    lines_gyro = gyro_path.read_text().splitlines()
    skip_gyro = lines_gyro.index("@DATA") + 1
    df_gyro = pd.read_csv(gyro_path, skiprows=skip_gyro, header=None, names=["t", "x", "y", "z"])
    df_gyro["t"] = df_gyro["t"] / 1e9
    df_gyro = df_gyro.drop_duplicates("t").sort_values("t")
    tg = df_gyro["t"].to_numpy()
    vg = df_gyro[["x", "y", "z"]].to_numpy(dtype=np.float64) * RAD2DEG  # rad/s -> deg/s

    # Overlapping time range
    t0 = max(ta[0], tg[0]) + 0.1  # trim 0.1s startup artifacts
    t1 = min(ta[-1], tg[-1])
    if t1 - t0 < (WIN / FS):
        return []

    # Resample onto 50 Hz uniform grid
    t_grid, v_acc = resample_uniform(ta, va, t_start=t0, t_end=t1, fs_target=FS)
    _, v_gyro = resample_uniform(tg, vg, t_start=t0, t_end=t1, fs_target=FS)

    min_len = min(len(v_acc), len(v_gyro))
    v_acc = v_acc[:min_len]
    v_gyro = v_gyro[:min_len]

    windows = []
    for start in range(0, min_len - WIN + 1, STRIDE):
        w_acc = v_acc[start:start + WIN]
        w_gyro = v_gyro[start:start + WIN]
        amag = np.sqrt(np.sum(w_acc**2, axis=1))

        windows.append({
            "dataset": "mobifall",
            "subject": f"mobi_{int(subj):02d}",
            "cohort": "adult",
            "activity": code,
            "trial": f"{code}_{subj}_{trial}",
            "start": start,
            "label": STABLE_WALK,
            "label_name": LABEL_NAMES[STABLE_WALK],
            "ax": w_acc[:, 0].tolist(),
            "ay": w_acc[:, 1].tolist(),
            "az": w_acc[:, 2].tolist(),
            "gx": w_gyro[:, 0].tolist(),
            "gy": w_gyro[:, 1].tolist(),
            "gz": w_gyro[:, 2].tolist(),
            "amag": amag.tolist(),
        })
    return windows


def main():
    print("=" * 70)
    print("STEP 1: HARMONIZE GAIT & STUMBLE DATASETS (50 Hz, 2.5s Windows)")
    print("=" * 70)

    all_windows = []

    # 1. Process SisFall trials
    sis_dataset = RAW_SISFALL / "SisFall_dataset"
    sis_enhanced = RAW_SISFALL / "SisFall_enhanced"
    print(f"Scanning SisFall files in: {sis_dataset}")

    sis_count = 0
    for act in SISFALL_ACTIVITIES:
        files = sorted(sis_dataset.rglob(f"{act}_*.txt"))
        print(f"  SisFall [{act}]: {len(files)} files found")
        for f in files:
            wins = process_sisfall_trial(f, sis_enhanced)
            all_windows.extend(wins)
            sis_count += 1
    print(f"Processed {sis_count} SisFall files -> {len(all_windows)} windows")

    # 2. Process MobiFall trials
    print(f"\nScanning MobiFall files in: {RAW_MOBIFALL}")
    mobi_count = 0
    mobi_start_idx = len(all_windows)
    for act in MOBIFALL_ACTIVITIES:
        files = sorted(RAW_MOBIFALL.rglob(f"{act}_acc_*.txt"))
        print(f"  MobiFall [{act}]: {len(files)} files found")
        for f in files:
            wins = process_mobifall_trial(f)
            all_windows.extend(wins)
            mobi_count += 1
    print(f"Processed {mobi_count} MobiFall files -> {len(all_windows) - mobi_start_idx} windows")

    df = pd.DataFrame(all_windows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "gait_windows.parquet"
    df.to_parquet(out_path, index=False)

    print("\n" + "=" * 70)
    print("DATA HARMONIZATION SUMMARY & ACCEPTANCE CHECKS")
    print("=" * 70)
    print(f"Total Windows: {len(df):,}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"Cohorts: {df['cohort'].value_counts().to_dict()}")
    print(f"Unique Subjects: {df['subject'].nunique()}")
    print(f"Labels: {df['label_name'].value_counts().to_dict()} (0: stable, 1: unstable)")

    # Gravity check on quiet windows
    amag_arr = np.array(df["amag"].tolist())
    amag_std = amag_arr.std(axis=1)
    amag_mean = amag_arr.mean(axis=1)
    still_mask = (amag_std < 0.05) & (df["label"] == 0).to_numpy()
    if still_mask.any():
        med_still_g = float(np.median(amag_mean[still_mask]))
        print(f"Gravity check on {still_mask.sum():,} still windows: median amag = {med_still_g:.3f} g [Expected: ~1.00g]")
    else:
        print("Note: No static/still windows in pure active walking set (expected for continuous walking).")

    print(f"\nSuccessfully wrote harmonized windows to:\n  -> {out_path}")


if __name__ == "__main__":
    main()
